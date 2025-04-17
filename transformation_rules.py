import time

from typing import Dict
from prompt_generators import *
from llm_utils import query_reasoning_model


def determine_transformation_rules(test_set: Dict, 
                              output_filepath: str = "results/final_approach/transformation_rules_new.json"):
    transformation_rules = {}

    for i, (task_name, task_content) in enumerate(test_set.items()):
        print(f"Querying reasoning model to determine transformation rule for task {task_name}")
        print(f"#{i +1} out of {len(test_set)}")

        messages = create_reasoning_model_prompt_new(task=task_content)

        try:
            transformation_rule, reasoning = query_reasoning_model(messages)
            print(f"####################\nTransformation rule found:\n{transformation_rule}\n\n")
            
            transformation_rules[task_name] = {} 
            transformation_rules[task_name]["transformation_rule"] = transformation_rule
            transformation_rules[task_name]["reasoning"] = reasoning
        except Exception as e:
            print(f"Error determining transformation rule {task_name}: {e}")
            print(f"Task {task_name} will be skipped.")
        
        # Save after every request
        try:
            with open(output_filepath, 'w') as f:
                json.dump(transformation_rules, f, indent=2)
            print(f"Saved transformation rules to {output_filepath}")
        except IOError as e:
            print(f"Error saving descriptions to {output_filepath}: {e}")
        
        # Prevent exceeding usage rates
        time.sleep(2)
    
    print(f"Finished processing tasks. All transformation rules saved to {output_filepath}")
    return transformation_rules


def load_transformation_rules(filepath: str = "results/final_approach/transformation_rules.json"):
    with open(filepath, 'r') as f:
        transformation_rules = json.load(f)
        print("\nLoaded Transformation Rules")
        
        return transformation_rules


def get_transformation_rule(task_id: str, rules_filepath: str = "results/final_approach/transformation_rules_full.json", delay: float = 0.005):
    def stream_text(text):
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()

    # Load the rules
    with open(rules_filepath, 'r') as f:
        rules = json.load(f)

    if task_id not in rules:
        print(f"Task ID '{task_id}' not found in transformation rules.\n")
        return None, None

    reasoning = rules[task_id].get("reasoning", "")
    transformation_rule = rules[task_id].get("transformation_rule", "")

    print(f"Reasoning for task {task_id}:\n")
    #stream_text(reasoning)
    print("Reasoning:\n" + reasoning)

    print(f"\nTransformation Rule for task {task_id}:\n")
    #stream_text(transformation_rule)
    print("Transformation rule:\n" + transformation_rule)

    return transformation_rule, reasoning 