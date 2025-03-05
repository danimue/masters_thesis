from utils import check_dataset, load_files_from_json, find_task, create_prompt, json_to_string, json_to_ascii
from llm_utils import generate_llm_response, initialize_client
from eval import extract_solution, evaluate_solution, fix_output_string, extract_code, execute_transform_function

import time
import json

import numpy as np
import pandas as pd

# from transformers import AutoTokenizer

def main() -> None:
    check_dataset()
    train, eval = load_files_from_json()
    
    test_set1 = dict(list(train.items())[0:10])
    
    client = initialize_client('mistral')
    
    # for task_index, (task_name, task_content) in enumerate(test_set1.items()):
        
    #     user_prompt = create_prompt(task_content, prompt_type="simple", reason=False)
    
    #     print(user_prompt)
    # results = []
    
    # code_results = []
    
    # # dont test with full train_set
    # test_set1 = dict(list(train.items())[16:25])
    # test_set2 = dict(list(train.items())[200:])
    # #test_set = train
    
    # ##########################################
    # #   Use this for oneshot program synthesis
    # ##########################################
    # for task_index, (task_name, task_content) in enumerate(test_set1.items()):
            
    #     print(f"Processing task {task_name} using program synthesis!")
    #     print("Generating code...")
        
    #     user_prompt = create_prompt(task_content, prompt_type="program_synthesis")
    #     system_prompt = "You are a very smart python programmer."
    
    #     # Generate 10 coding solutions for one task
    #     # will return a list, as multiple samples are drawn
        
    #     responses_list = []
    #     #temperatures = [0.2, 0.4, 0.6, 0.8]
    #     temperatures = [0.2]
        
    #     for temp in temperatures:
    #         print(f"Generating solutions using temperature {temp}")
    #         responses_list.extend(generate_llm_response(user_prompt, system_prompt, num_samples=10, temperature=temp))
        
    #     if responses_list:
    #         print("Number of responses generated: " + str(len(responses_list)))
        
    #     # From each response in the list, extract the python code and save to another list   
    #     generated_programs = []    
    #     for response in responses_list:
    #         extracted_code = extract_code(response)
    #         generated_programs.append(extracted_code)
        
    #     print(f"Saved {len(generated_programs)} from output.")
        
    #     for i, generated_code in enumerate(generated_programs):
        
    #         print(f"Attempting to run program #{i}...")
        
    #         dicts_list = []

    #         # Set up individual input/output examples as arrays
    #         for example in task_content['train']:
    #             example_input = example['input']
    #             example_output = np.array(example['output'])
                
    #             # Then execute generated code for each example
    #             output_grid_with_llm_code = execute_transform_function(generated_code, example_input)

    #             try:
    #                 solution_array = np.array(output_grid_with_llm_code)
    #             except:
    #                 print("No numpy array could be made :(")

    #             # compare generated output to actual output
    #             result = evaluate_solution(solution_array, example_output)
    #             dicts_list.append(result)
        
    #         for d in dicts_list:
    #             print(d)
            
    #         percentage_correct_values = [d['percentage_correct'] for d in dicts_list]
    #         average_percentage_correct = sum(percentage_correct_values) / len(percentage_correct_values)
        
    #         count_perfect_scores = 0   
    #         for d in dicts_list:
    #             if d['percentage_correct'] == 1.0:
    #                 count_perfect_scores += 1
                
    #         print(i, average_percentage_correct, count_perfect_scores)

    #         # try to save results:
            
    #         row = (
    #         {
    #             'task': task_name,
    #             'sample_num': i+1,
    #             'extracted_code': generated_code,
    #             'average_percentage_correct': average_percentage_correct,
    #             'count_perfect_scores': count_perfect_scores
    #         })
            
    #         code_results.append(row)
            
    #     df = pd.DataFrame(code_results)
    #     df.to_csv("results/codestral_test_results_temp04.csv", index=False)
            
    #     time.sleep(1.5)

        

    
    #######################################
    #   Used this for oneshot simple prompt
    #######################################
    results = []
    
    for i, (task_name, task_content) in enumerate(test_set1.items()):
    
        print(f"Processing task {task_name}")
        
        user_prompt = create_prompt(task_content, prompt_type='simple', reason=False)
        system_prompt = "You are a very smart puzzle solver."
        
        print("Asking llm for answer!")
        response = generate_llm_response(user_prompt, system_prompt, client=client, num_samples=1, temperature=0.0)
        
        print(response)
        
        print("Evaluating answer")
        
        try:
            
            extracted_answer = extract_solution(response)
            extracted_answer = np.array(fix_output_string(extracted_answer))
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
        time.sleep(1.5)
        
        if (i + 1) % 20 == 0:
            df = pd.DataFrame(results)
            df.to_csv("results/mistral_small_test.csv", index=False)
            print(f"Data saved to CSV after {i + 1} iterations.")
    
    #qwen-72b-vl_oneshot_reason_p1    
    df = pd.DataFrame(results)
    df.to_csv("results/mistral_small_test.csv", index=False)
    print("Data saved to CSV at the end of the loop.")

if __name__ == "__main__":
    main()
    
# TODO: 
# Loop over entire train set
# Save results somewhere for eval
# Add no result found in answer for eval possibilities
# Save prompt text in file
# Figure out temperature setting etc.