from utils import *
from prompt_generators import create_basic_prompt, generate_coding_prompt, generate_prompt_with_image, generate_code_fixing_prompt, create_reasoning_model_prompt
from llm_utils import generate_llm_response, initialize_client
from eval import extract_solution, evaluate_solution, convert_output_to_array, extract_code, execute_transform_function

import os

import ast
import time
import json
import asyncio

import numpy as np
import pandas as pd


#######################################
#       Please dont run this
#       at the moment! Might 
#       overwrite important stuff.
######################################
def main() -> None:
        
    check_dataset()
    train, eval = load_files_from_json()
    
    # config = {
    #     'model': 'codestral-latest',
    #     'solution_type': 'program_synthesis',
    #     'temperatures': [0.1, 0.3, 0.5, 0.7, 0.9],
    #     'samples_per_step': 10,
    #     'code_correction_steps': 4
    # }
    
    config = {
        #'model': 'mistral-large-latest',
        #'model': 'meta-llama/Llama-Vision-Free',
        #'model': 'pixtral-large-latest',
        #'model': 'qwen/qwen2.5-vl-72b-instruct:free',
        #'model': 'google/gemma-3-27b-it:free',
        #'model': 'codestral-latest',
        # 'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
        #'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
        #'model': 'qwen/qwq-32b:free',
        #'model': 'qwen2.5-14b-instruct',
        #'model': 'qwen2.5-32b-instruct',
        #'model': 'qwen2.5-72b-instruct',
        #'model': 'deepseek/deepseek-r1-distill-llama-70b:free',
        #'model': 'mistral-small-latest',
        'model': 'qwq-32b',
        'solution_type': 'simple',
        'temperatures': [0.1, 0.3, 0.5, 0.7, 0.9],
        'samples_per_step': 20,
        'code_correction_steps': 4
    }
    
    client = initialize_client('alibaba')
    
    #client = initialize_client('dashscope')
    #client = initialize_client('openrouter')
    print("Client initialized!")
    
    
    ############################
    ##  Code Fixing
    ############################

    # df = pd.read_csv("results/program_synthesis/codestral_full.csv")
    
    # solved_tasks = set(df.query("average_percentage_correct == 1.0")['task'].to_list())
    # relevant_tasks = set(df.query("average_percentage_correct > 0.8 and average_percentage_correct != 1 and task not in @solved_tasks")["task"].to_list())
    
    # attempt_fixing = (
    #     df
    #     .query("""task in @relevant_tasks and \
    #             average_percentage_correct != 1 and \
    #             (average_percentage_correct > 0.8 or \
    #             count_perfect_scores > 1)""")
    #     .drop_duplicates(subset=["average_percentage_correct"], keep='first')
    #     .sort_values(["count_perfect_scores", "average_percentage_correct"], ascending=False)
    #     .groupby("task")
    #     .head(3)
    # )
    
    
    # additional_infos = []
    # code_results = []

    # for index, row in attempt_fixing.iterrows():

    #     print(f"Attempting to fix code for {row['task']}.")
    #     print(f"#{index} out of {attempt_fixing.shape[0]}")

    #     print("Generating code...")
    #     responses_list = []
    #     additional_info = []

    #     task_id = row['task']
    #     task = train[task_id]
    #     answer = row['full_answer']
    #     generated_grids = ast.literal_eval(row['generated_grids'])

    #     messages = (generate_code_fixing_prompt(task=task, 
    #                                           extracted_answer=answer, 
    #                                           generated_inputs=generated_grids))

    #     for temp in config['temperatures']:
    #         print(f"Generating {config['samples_per_step']} solutions using temperature {temp}")
    #         responses, additional_info = generate_llm_response(messages, 
    #                                                            client=client, 
    #                                                            model=config['model'],
    #                                                            num_samples=config['samples_per_step'], 
    #                                                            temperature=temp)
            
    #         # make sure we dont exceed usage rates
    #         time.sleep(1.5)

    #         # Save the responses text to extract code and test its performance
    #         responses_list.extend(responses)

    #         # Save additional info for later data analysis
    #         additional_info.insert(0, task_id)
    #         additional_info.append(temp)
    #         additional_infos.append(additional_info)

    #     # From each response in the list, extract the python code and save to another list   
    #     generated_programs = []    
    #     for response in responses_list:
    #         extracted_code = extract_code(response)
    #         generated_programs.append(extracted_code)

    #     print(f"Saved {len(generated_programs)} from output.")

    #     generated_output_grids = []

    #     for i, generated_code in enumerate(generated_programs):
        
    #         print(f"Attempting to run program #{i}...")
    #         generated_output_grids = []
    #         dicts_list = []
    #         # Set up individual input/output examples as arrays
    #         for example in task['train']:
    #             example_input = example['input']
    #             example_output = np.array(example['output'])

    #             # Then execute generated code for each example
    #             output_grid_with_llm_code = execute_transform_function(generated_code, example_input)
    #             generated_output_grids.append(output_grid_with_llm_code)

    #             try:
    #                 solution_array = np.array(output_grid_with_llm_code)
    #             except:
    #                 print("No numpy array could be made :(")
    #             # compare generated output to actual output
    #             result = evaluate_solution(solution_array, example_output)
    #             dicts_list.append(result)

    #         for d in dicts_list:
    #             print(d)

    #         # calculate performance indicators for each code generation
    #         percentage_correct_values = [d['percentage_correct'] for d in dicts_list]
    #         average_percentage_correct = sum(percentage_correct_values) / len(percentage_correct_values)
    #         count_perfect_scores = sum(1 for d in dicts_list if d['percentage_correct'] == 1.0)

    #         print(i, average_percentage_correct, count_perfect_scores)
    #         # save results:            
    #         row = (
    #         {
    #             'task': task_id,
    #             'sample_num': i+1,
    #             'full_answer': responses_list[i],
    #             'extracted_code': generated_code,
    #             'generated_grids': generated_output_grids,
    #             'average_percentage_correct': average_percentage_correct,
    #             'count_perfect_scores': count_perfect_scores
    #         })

    #         code_results.append(row)

    #     # Save results after every loop in case something breaks    
    #     df_code_fix = pd.DataFrame(code_results)
    #     df_code_fix.to_csv("results/program_synthesis/codestral_code_fixing_test.csv", index=False)

    #     df_info = pd.DataFrame(additional_infos,
    #                        columns=['task_name', 'processing_time', 'samples_drawn', 'input_tokens', 'output_tokens', 'temperature'])
    #     df_info.to_csv("results/program_synthesis/codestral_code_fixing_test_info.csv", index=False)

    ###################################
    ##   Visual Reasoning
    ###################################    
    # Create smaller dataset for testing purposes
    # test_set = dict(list(train.items())[65:])
    
    # results = []

    # for task_index, (task_name, task_content) in enumerate(test_set.items()):
    #     print(f"Processing task {task_name} using visual reasoning.")
    #     print(f"Task #{task_index + 1} out of {len(test_set)}")
        
    #     messages = generate_prompt_with_image(task_name, task_content)

    #     response = generate_llm_response(messages=messages, 
    #                                      client=client, 
    #                                      model=config['model'])
    #     print("Evaluating answer")
        
    #     extracted_answer = None     # Initialize to None before the try block
    #     try:
    #         extracted_answer_temp = extract_solution(response)            
    #         extracted_answer = convert_output_to_array(extracted_answer_temp)
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
    #     time.sleep(2)

    #     df = pd.DataFrame(results)
    #     df.to_csv("results/vl_models/mistral-small_visual-reasoning_p3.csv", index=False)
    #     print(f"Data saved to CSV after {task_index + 1} iterations.")
    
    # # save at the very end
    # df = pd.DataFrame(results)
    # df.to_csv("results/vl_models/mistral-small_visual-reasoning_p3.csv", index=False)
    # print("Data saved to CSV at the end of the loop.")
    

    ###################################
    ##   Use this for program synthesis
    ###################################
    
    ####
    # only the tasks that qwen-qwq could not solve!
    # qwen = pd.read_csv("results/qwen-qwq-32b_reasoning-updated.csv")
    # qwen_solved = set(qwen.query("percentage_correct == 1")['task'].to_list())

    # test_set = {
    #   key: value
    #   for key, value in train.items()
    #   if key not in qwen_solved
    # }
    
    # print(len(test_set))
    
    # additional_infos = []
    # code_results = []

    # # Start looping over tasks!    
    # for task_index, (task_name, task_content) in enumerate(test_set.items()):
            
    #     print(f"Processing task {task_name} using program synthesis.")
        
        
        
    #     # prompt = generate_coding_prompt(task_content)
    #     # messages = [prompt.copy() for _ in range(config['samples_per_step'])]
        
    #     messages = generate_coding_prompt(task_content)
        
    #     print("Generating code...")
    #     responses_list = []
    #     additional_info = []
    #     temperatures = config['temperatures']
        
    #     for temp in temperatures:
    #         print(f"Generating {config['samples_per_step']} solutions using temperature {temp}")
    #         responses, additional_info = generate_llm_response(messages, client=client, model=config['model'],
    #                                                 num_samples=config['samples_per_step'], temperature=temp)
            
    #         # Save the responses text to extract code and test its performance
    #         responses_list.extend(responses)
            
    #         # Save additional info for later data analysis
    #         additional_info.insert(0, task_name)
    #         additional_info.append(temp)
    #         additional_infos.append(additional_info)
        
    #     # From each response in the list, extract the python code and save to another list   
    #     generated_programs = []    
    #     for response in responses_list:
    #         extracted_code = extract_code(response)
    #         generated_programs.append(extracted_code)
        
    #     print(f"Saved {len(generated_programs)} from output.")
        
    #     #generated_output_grids = []
        
    #     for i, generated_code in enumerate(generated_programs):
        
    #         print(f"Attempting to run program #{i}...")

    #         generated_output_grids = []
    #         dicts_list = []

    #         # Set up individual input/output examples as arrays
    #         for example in task_content['train']:
    #             example_input = example['input']
    #             example_output = np.array(example['output'])
                
    #             # Then execute generated code for each example
    #             output_grid_with_llm_code = execute_transform_function(generated_code, example_input)
    #             generated_output_grids.append(output_grid_with_llm_code)
                
    #             try:
    #                 solution_array = np.array(output_grid_with_llm_code)
    #             except:
    #                 print("No numpy array could be made :(")

    #             # compare generated output to actual output
    #             result = evaluate_solution(solution_array, example_output)
    #             dicts_list.append(result)
        
    #         for d in dicts_list:
    #             print(d)
            
    #         # calculate performance indicators for each code generation
    #         percentage_correct_values = [d['percentage_correct'] for d in dicts_list]
    #         average_percentage_correct = sum(percentage_correct_values) / len(percentage_correct_values)
    #         count_perfect_scores = sum(1 for d in dicts_list if d['percentage_correct'] == 1.0)
            
    #         print(i, average_percentage_correct, count_perfect_scores)

    #         # save results:            
    #         row = (
    #         {
    #             'task': task_name,
    #             'sample_num': i+1,
    #             'full_answer': responses_list[i],
    #             'extracted_code': generated_code,
    #             'generated_grids': generated_output_grids,
    #             'average_percentage_correct': average_percentage_correct,
    #             'count_perfect_scores': count_perfect_scores,
    #             'generation_step': 0
    #         })
            
    #         code_results.append(row)
        
    #     # Save results after every loop in case something breaks    
    #     df = pd.DataFrame(code_results)
    #     df.to_csv("results/program_synthesis/qwen-32b_code_p1.csv", index=False)
        
    #     df2 = pd.DataFrame(additional_infos,
    #                        columns=['task_name', 'processing_time', 'samples_drawn', 'input_tokens', 'output_tokens', 'temperature'])
    #     df2.to_csv("results/program_synthesis/qwen-32b_code_info_p1.csv", index=False)
        
    #     # Take a tiny break in order to not exceed token limits
    #     # Might not really be necessary as prompts take a long time to process anyways    
    #     time.sleep(1.5)
    
    #######################################
    #   Used this for oneshot simple prompt
    #######################################
    test_set = dict(list(train.items())[:1])
    
    results = []
    
    for i, (task_name, task_content) in enumerate(test_set.items()):
    
        print(f"Processing task {task_name}")
        print(f"#{i +1} out of {len(test_set)}")
        
        #messages = create_reasoning_model_prompt(task=task_content)
        messages = create_reasoning_model_prompt(task=task_content)
        
        print("Asking llm for answer!")
        response = generate_llm_response(messages=messages, 
                                         client=client,
                                         model=config['model'],
                                         num_samples=1,
                                         temperature=0.6)
        
        print("Evaluating answer")
        
        extracted_answer = None     # Initialize to None before the try block
        try:
            extracted_answer_temp = extract_solution(response)            
            extracted_answer = convert_output_to_array(extracted_answer_temp)
        except ValueError as e:
            print(f"Error converting extracted solution to NumPy array: {e}")
            extracted_answer = None
        
        correct_solution = np.array(task_content['test'][0]['output'])
        result = evaluate_solution(extracted_answer, correct_solution)
        
        print(result)
        
        row = (
            {
                'task': task_name,
                'llm_full_answer': response,
                'llm_extracted_answer': extracted_answer
            })
        
        row.update(result)
        results.append(row)
        
        print("Waiting for one second...")
        time.sleep(3)
        
        
        df = pd.DataFrame(results)
        df.to_csv("dont_run!", index=False)
        print(f"Data saved to CSV after {i + 1} iterations.")
     
    df = pd.DataFrame(results)
    df.to_csv("dont_run!", index=False)
    print("Data saved to CSV at the end of the loop.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Final processing time for all tasks: {end_time - start_time}")