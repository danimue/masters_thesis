import json
import time

from prompt_generators import generate_image_description_prompt
from llm_utils import generate_llm_response
from typing import Dict

def generate_vlm_descriptions(test_set: Dict, 
                              client, 
                              model,
                              output_filepath: str = "results/final_approach/vlm_descriptions.json"):
    vlm_descriptions = {}

    for i, (task_name, task_content) in enumerate(test_set.items()):
        print(f"Generating image description for task {task_name}")
        print(f"#{i +1} out of {len(test_set)}")

        messages = generate_image_description_prompt(task_id=task_name, task=task_content)

        try:
            response = generate_llm_response(messages=messages, 
                                             client=client,
                                             model=model,
                                             num_samples=1,
                                             temperature=0)
            print(f"Generated image descriptions:\n{response}")
            vlm_descriptions[task_name] = response
        except Exception as e:
            print(f"Error generating description for task {task_name}: {e}")
            print(f"Description for {task_name} will be skipped.")
        
        # Save after every request
        try:
            with open(output_filepath, 'w') as f:
                json.dump(vlm_descriptions, f, indent=2)
            print(f"Saved descriptions to {output_filepath}")
        except IOError as e:
            print(f"Error saving descriptions to {output_filepath}: {e}")

        vlm_descriptions[task_name] = response
        
        # Prevent exceeding usage rates
        time.sleep(2)
    
    print(f"Finished processing (or attempting) VLM descriptions. All results saved to {output_filepath}")
    return vlm_descriptions


def load_vlm_descriptions(filepath: str = "results/final_approach/vlm_descriptions.json"):
    with open(filepath, 'r') as f:
        vlm_descriptions = json.load(f)
        print("\nLoaded VLM Descriptions:")
        
        return vlm_descriptions