from prompt_generators import create_basic_prompt, generate_coding_prompt, generate_prompt_with_image, generate_code_fixing_prompt, create_reasoning_model_prompt, generate_continued_code_fixing_prompt
from llm_utils import generate_llm_response, initialize_client, generate_llm_response_async
from eval import extract_solution, evaluate_solution, convert_output_to_array, extract_code, execute_transform_function

import ast
import time
import json
import asyncio

import numpy as np
import pandas as pd

async def generate_code_samples(task, messages) -> None:
    
    config = {
        'coding_model':'qwen/qwen-2.5-coder-32b-instruct',
        'solution_type': 'program_synthesis',
        'temperatures': [0.1, 0.3, 0.5, 0.7, 0.9],
        'samples_per_step': 10,
        'code_correction_steps': 4
    }
    
    # Async client is necessary as Alibaba Cloud AND Openrouter do not support n-samples parameter.
    # Instead of using n-parameter, I simply send n requests at the same time, which has the same effect.
    # The only API that reliably supported that was Mistral, which was not used in the final approach
    client = initialize_client('openrouter_async')
    print("Client initialized!")
    
    code_results = []
    sample_counter = 1
    temperatures = config['temperatures']
    n = config['samples_per_step']
    
    
    
    # start the generation loop
    print(f"Attempting to solve task {task} through program synthesis...")
    for temp in temperatures:
        print(f"Generating {config['samples_per_step']} solutions using temperature {temp}")
        
        tasks = [generate_llm_response_async(client=client, 
                                             model=config['coding_model'], 
                                             temperature=temp, 
                                             messages=messages) 
                     for _ in range(n)]
        results = await asyncio.gather(*tasks)
         
        
        # Check if good result was found, so we can stop generating programs
        # Extract code from LLM answers:
        generated_programs = []    
        for result in results:
            extracted_code = extract_code(result)
            generated_programs.append(extracted_code)
        print(f"Saved {len(generated_programs)} from output.")
        
        # Then run the code generated using exec() -- not best practice! 
        # But easier to set up than sandboxing
        for i, generated_code in enumerate(generated_programs):
            print(f"Attempting to run program #{i}...")
            generated_output_grids = []
            dicts_list = []
            
            # Set up individual input/output examples as arrays
            for example in task['train']:
                example_input = example['input']
                example_output = np.array(example['output'])
                # Then execute generated code for each example
                output_grid_with_llm_code = execute_transform_function(generated_code, example_input)
                generated_output_grids.append(output_grid_with_llm_code)
                try:
                    solution_array = np.array(output_grid_with_llm_code)
                except:
                    print("No numpy array could be made :(")
                # Compare generated output to actual output
                result = evaluate_solution(solution_array, example_output)
                dicts_list.append(result)
            for d in dicts_list:
                print(d)
    
            # Calculate evaluation for each code generation
            percentage_correct_values = [d['percentage_correct'] for d in dicts_list]
            average_percentage_correct = sum(percentage_correct_values) / len(percentage_correct_values)
            count_perfect_scores = sum(1 for d in dicts_list if d['percentage_correct'] == 1.0)
            print("Code sample: " + str(i), 
                  "Percentage correct: " + str(round(average_percentage_correct, 2)),
                  "Num perfect scores: " + str(round(count_perfect_scores, 2)))
            
            # Add to results list          
            row = (
            {
                'task': task,
                'sample_num': sample_counter,
                'full_answer': results[i],
                'extracted_code': generated_code,
                'generated_grids': generated_output_grids,
                'average_percentage_correct': average_percentage_correct,
                'count_perfect_scores': count_perfect_scores,
                'temperature': temp,
                'generation_step': 0
            })
            code_results.append(row)
            sample_counter += 1
        
        # Save to dataframe and csv  
        df = pd.DataFrame(code_results)
        df.to_csv("results/demo_solutions.csv", index=False)
            
        # Check if solution found:
        if not (df.query("task == @task and average_percentage_correct == 1").empty):
            print("Correct solution found, moving to next task.")
            solution_found = True
            break
        else:
            print("No correct solution found in this batch.")
            
    return df


async def fix_code_with_llm(attempt_fixing, task, client) -> None:
    
    config = {
        'model': 'qwen/qwen-2.5-coder-32b-instruct',
        'solution_type': 'code_fixing',
        'temperatures': [0.5],
        'samples_per_step': 5,
        'code_correction_steps': 2
    }
    
    #Variables for final results
    df_code_fix = pd.DataFrame()

    #Run the code fixing for every entry in the dataframe that seems promising, max 3 per task
    for index, (_, row) in enumerate(attempt_fixing.iterrows()):

        task_id = row['task']
        answer = row['full_answer']
        generated_grids = row['generated_grids']
        num_requests = config['samples_per_step']
        model = config['model']

        messages = (generate_code_fixing_prompt(task=task, 
                                              extracted_answer=answer, 
                                              generated_inputs=generated_grids))

        for attempt_counter in range(0, config['code_correction_steps']):
        
            responses_list = []
            code_results = []
            
            print(f"Attempting to fix code for {row['task']}.")
            print(f"#Attempting to fix program {index+1} out of {attempt_fixing.shape[0]} -- Attempt #{attempt_counter+1} out of {config['code_correction_steps']}.")
            print("Generating code...")

            for temp in config['temperatures']:
                print(f"Generating {config['samples_per_step']} solutions using temperature {temp}")

                tasks = [generate_llm_response_async(client=client, model=model, temperature=temp, messages=messages) 
                         for _ in range(num_requests)]
                results = await asyncio.gather(*tasks)

                # Save the responses text to extract code and test its performance
                responses_list.extend(results)

            # From each response in the list, extract the python code and save to another list   
            generated_programs = []    
            for response in responses_list:
                extracted_code = extract_code(response)
                generated_programs.append(extracted_code)

            print(f"Saved {len(generated_programs)} from output.")

            generated_output_grids = []

            for i, generated_code in enumerate(generated_programs):
                print(f"Attempting to run program #{i}...")

                generated_output_grids = []
                dicts_list = []
                # Set up individual input/output examples as arrays
                for example in task['train']:
                    example_input = example['input']
                    example_output = np.array(example['output'])
                    # Then execute generated code for each example
                    output_grid_with_llm_code = execute_transform_function(generated_code, example_input)
                    generated_output_grids.append(output_grid_with_llm_code)
                    try:
                        solution_array = np.array(output_grid_with_llm_code)
                    except:
                        print("No numpy array could be made :(")
                    # compare generated output to actual output
                    result = evaluate_solution(solution_array, example_output)
                    dicts_list.append(result)

                for d in dicts_list:
                    print(d)

                # calculate performance indicators for each code generation
                percentage_correct_values = [d['percentage_correct'] for d in dicts_list]
                average_percentage_correct = sum(percentage_correct_values) / len(percentage_correct_values)
                count_perfect_scores = sum(1 for d in dicts_list if d['percentage_correct'] == 1.0)

                print(i, average_percentage_correct, count_perfect_scores)

                # save results:            
                row = (
                {
                    'task': task_id,
                    'sample_num': i+1,
                    'full_answer': responses_list[i],
                    'extracted_code': generated_code,
                    'generated_grids': generated_output_grids,
                    'average_percentage_correct': average_percentage_correct,
                    'count_perfect_scores': count_perfect_scores,
                    'num_attempt': attempt_counter
                })
                code_results.append(row)

            partial_df = pd.DataFrame(code_results)
            df_code_fix = pd.concat([df_code_fix, partial_df], ignore_index=True)
        
            # Check if solution was found:
            if (not partial_df.query("average_percentage_correct == 1").empty):
                print(f"Solution found after {attempt_counter} attempts!")
                break
            
                    # Else determine best generated solution and start new code fixing attempt
            else:
                sorted_df = partial_df.sort_values(["count_perfect_scores", "average_percentage_correct"], ascending=False).head(1)
                result_dict = sorted_df.to_dict(orient='records')[0]

                # Generate new code fixing prompt
                updated_user_prompt = generate_continued_code_fixing_prompt(result_dict['generated_grids'])

                # Update "conversation" with LLM
                messages.append({"role": "assistant", "content": result_dict['full_answer']})
                messages.append({"role": "user", "content": updated_user_prompt})

                print("Starting new attempt...")
                attempt_counter += 1
    
        # Save results after every loop in case something breaks        
        df_code_fix.to_csv("results/demo_solutions_code_fixing.csv", index=False)
        
    return df_code_fix