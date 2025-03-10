from utils import check_dataset, load_files_from_json, find_task, create_prompt, generate_coding_prompt
from llm_utils import generate_llm_response, initialize_client
from eval import extract_solution, evaluate_solution, fix_output_string, extract_code, execute_transform_function

import time
import json

import numpy as np
import pandas as pd


def main() -> None:
        
    check_dataset()
    train, eval = load_files_from_json()
    
    config = {
        'model': 'codestral-latest',
        'solution_type': 'program_synthesis',
        'temperatures': [0.1, 0.3, 0.5, 0.7, 0.9],
        'samples_per_step': 10,
        'code_correction_steps': 4
    }
    
    # Initializing client:
    client = initialize_client('mistral')
    print("Client initialized!")
    
    # Create smaller dataset for testing purposes
    test_set = dict(list(train.items())[0:100])
    
    ###################################
    ##   Use this for program synthesis
    ###################################
    
    additional_infos = []
    code_results = []

    # Start looping over tasks!    
    for task_index, (task_name, task_content) in enumerate(test_set.items()):
            
        print(f"Processing task {task_name} using program synthesis!")
        
        user_prompt = generate_coding_prompt(task_content)
        system_prompt = """You are a very talented python programmer. You will be given multiple paired example input and outputs. The outputs were produced by applying a transformation rule to the inputs. Your task is to determine the transformation rule and implement it in code. The inputs and outputs are each 'grids'. A grid is a rectangular matrix of integers between 0 and 9 (inclusive). The integer values represent colors. 
        The transformation rule that you have to deduce might have multiple components and can be fairly complex. In your reasoning you will break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating your overall conclusion. This will avoid large leaps in reasoning. You reason in detail and for as long as is necessary to fully determine the transformation rule."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        
        print("Generating code...")
        responses_list = []
        additional_info = []
        temperatures = config['temperatures']
        
        for temp in temperatures:
            print(f"Generating {config['samples_per_step']} solutions using temperature {temp}")
            responses, additional_info = generate_llm_response(messages, client=client, model=config['model'],
                                                    num_samples=config['samples_per_step'], temperature=temp)
            
            # Save the responses text to extract code and test its performance
            responses_list.extend(responses)
            
            # Save additional info for later data analysis
            additional_info.insert(0, task_name)
            additional_info.append(temp)
            additional_infos.append(additional_info)
        
        # From each response in the list, extract the python code and save to another list   
        generated_programs = []    
        for response in responses_list:
            extracted_code = extract_code(response)
            generated_programs.append(extracted_code)
        
        print(f"Saved {len(generated_programs)} from output.")
        
        #generated_output_grids = []
        
        for i, generated_code in enumerate(generated_programs):
        
            print(f"Attempting to run program #{i}...")

            generated_output_grids = []
            dicts_list = []

            # Set up individual input/output examples as arrays
            for example in task_content['train']:
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
                'task': task_name,
                'sample_num': i+1,
                'full_answer': responses_list[i],
                'extracted_code': generated_code,
                'generated_grids': generated_output_grids,
                'average_percentage_correct': average_percentage_correct,
                'count_perfect_scores': count_perfect_scores
            })
            
            code_results.append(row)
        
        # Save results after every loop in case something breaks    
        df = pd.DataFrame(code_results)
        df.to_csv("results/program_synthesis/codestral_code_p1.csv", index=False)
        
        df2 = pd.DataFrame(additional_infos,
                           columns=['task_name', 'processing_time', 'samples_drawn', 'input_tokens', 'output_tokens', 'temperature'])
        df2.to_csv("results/program_synthesis/codestral_code_info_p1.csv", index=False)
        
        # Take a tiny break in order to not exceed token limits
        # Might not really be necessary as prompts take a long time to process anyways    
        time.sleep(1.5)

        

    
    #######################################
    #   Used this for oneshot simple prompt
    #######################################
    # results = []
    
    # for i, (task_name, task_content) in enumerate(train.items()):
    
    #     print(f"Processing task {task_name}")
        
    #     user_prompt = create_prompt(task_content, prompt_type='simple', reason=False)
    #     system_prompt = "You are a very smart puzzle solver."
        
    #     print("Asking llm for answer!")
    #     response = generate_llm_response(user_prompt, system_prompt, client=client, num_samples=1, temperature=0.0)
        
    #     print(response)
        
    #     print("Evaluating answer")
        
    #     extracted_answer = None     # Initialize to None before the try block
    #     try:
    #         extracted_answer_temp = extract_solution(response)            
    #         extracted_answer_temp = fix_output_string(extracted_answer_temp)
    #         if extracted_answer_temp is not None:
    #             extracted_answer = np.array(extracted_answer_temp)
    #     except ValueError as e:
    #         print(f"Error converting extracted solution to NumPy array: {e}")
    #         extracted_answer = None
        
    #     correct_solution = np.array(task_content['test'][0]['output'])
    #     result = evaluate_solution(extracted_answer, correct_solution)
        
    #     print(result)
        
    #     row = (
    #         {
    #             'task': task_name,
    #             'llm_full_answer': response,
    #             'llm_extracted_answer': extracted_answer
    #         })
        
    #     row.update(result)
    #     results.append(row)
        
    #     print("Waiting for one second...")
    #     time.sleep(1.5)
        
    #     if (i + 1) % 10 == 0:
    #         df = pd.DataFrame(results)
    #         df.to_csv("results/llama-11b-vl_noreason.csv", index=False)
    #         print(f"Data saved to CSV after {i + 1} iterations.")
    
    # #llama-11b-vl_noreason  
    # df = pd.DataFrame(results)
    # df.to_csv("results/llama-11b-vl_noreason.csv", index=False)
    # print("Data saved to CSV at the end of the loop.")

if __name__ == "__main__":
    main()