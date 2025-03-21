import os
import json
import requests
import zipfile
import shutil
import base64

from typing import Any, Dict

def check_dataset():
    """
    Checks if ARC-AGI dataset exists and downloads it if not.
    """

    # I sometimes have trouble with the working directory not being the same as the script directory.
    # Therefore I include this little bit, changing cwd to script directory.
    # Otherwise I would not be able to use relative paths.

    # print("Current working directory:", os.getcwd())
    # print("Script location:", os.path.abspath(__file__))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != script_dir:
        os.chdir(script_dir)
        print("Working directory set to:", os.getcwd())

    # Check if directory for ARC-data already exists
    # There should be 800 json files
    if len([file for path, directories, files in os.walk("data") for file in files]) == 800:
        print("ARC data complete.")

    else:
        if not os.path.isdir("data"):
            print("Creating new directory...")
            #os.mkdir("data")

        # Download zip file
        zip_url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
        zip_path = "ARC-AGI-master.zip"

        response = requests.get(zip_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download repository: {response.status_code}, {response.text}")

        with open(zip_path, "wb") as f:
            f.write(response.content)
            print(f"Repository downloaded as {zip_path}")

        with zipfile.ZipFile(zip_path) as archive:
            for file in archive.namelist():
                if file.startswith('ARC-AGI-master/data/'):
                    archive.extract(file)

        # Define the source and destination paths
        source_dir = './ARC-AGI-master/data/'
        destination_dir = './'  # Move to root directory as 'data/'

        # Move the entire 'data' folder
        if os.path.exists(source_dir):
            shutil.move(source_dir, destination_dir)

        # Clean up the zip file and leftover extraction file
        os.remove(zip_path)
        os.rmdir('./ARC-AGI-master/')
        print(f"Cleaned up zip file and leftover extraction file.")

        print("Success!")


def load_files_from_json() -> tuple:
    """
    Loads JSON files from the 'training' and 'evaluation' ARC data.

    Returns:
        tuple: A tuple containing two dictionaries:
            - train (dict): Data from the training directory.
            - eval (dict): Data from the evaluation directory.
    """
    train = {}
    eval = {}

    # Directories for dataset
    training_data_dir = os.path.abspath("data/training")
    eval_data_dir = os.path.abspath("data/evaluation")

    # Load training
    for file_name in os.listdir(training_data_dir):
        file_path = os.path.join(training_data_dir, file_name)

        with open(file_path, "r") as f:
            task_name = file_name.split(".")[0]
            train[task_name] = json.load(f)

    # Load eval
    for file_name in os.listdir(eval_data_dir):
        file_path = os.path.join(eval_data_dir, file_name)

        with open(file_path, "r") as f:
            task_name = os.path.splitext(file_name)[0]  # Remove file extension
            eval[task_name] = json.load(f)

    return train, eval


# converts dict to a nicely formatted string for prompt creation
def json_to_string(task: dict, include_test: bool = False) -> str:
    """
    Converts a given ARC-task from its json-like form to a string, so that it can be used in a prompt.

    Parameters:
        - task (dict): the task from train or eval set
        - include_test (bool): by default, the test case will not be included
    """

    train_tasks = task["train"]
    test_task = task["test"]

    # Training Examples
    final_output = "Training Examples\n"
    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n["
        for index, row in enumerate(task["input"]):
            if index == len(task["input"])-1:
                final_output += f"\n{str(row)}"
            else:
                final_output += f"\n{str(row)},"
        final_output += "]\n\n"
        final_output += f"Example {i + 1}: Output\n["
        for index, row in enumerate(task["output"]):
            if index == len(task["output"])-1:
                final_output += f"\n{str(row)}"
            else:
                final_output += f"\n{str(row)},"
        final_output += "]\n\n"

    # Test Example
    if include_test:
        final_output += "Test\n["
        for index, row in enumerate(test_task[0]["input"]):
            if index == len(test_task[0]["input"])-1:
                final_output += f"\n{str(row)}"
            else:
                final_output += f"\n{str(row)},"
        final_output += "]"

    return final_output


def json_to_ascii(task: dict[str, Any], include_test: bool = False) -> str:
    
    result_str = "Training examples as character representation:\n"
    
    # TODO: come up with better letters? maybe b for black etc.
    letter_mapping = {0: ' ', 1: 'b', 2: 'c', 3: 'd', 4: 'e',
                      5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}
    
    # Training examples
    for i, element in enumerate(task["train"]):        
        result_str += f"Example {i + 1}: Input\n"
        
        # Input
        for line in element["input"]:
            for index, int in enumerate(line):
                if index == len(line) -1:
                    result_str += letter_mapping[int]
                else:
                    result_str += letter_mapping[int] + "|"
            result_str += "\n"
        result_str += "\n\n"
        
        # Output
        result_str += f"Example {i + 1}: Output\n"
        for line in element["output"]:
            for index, int in enumerate(line):
                if index == len(line) -1:
                    result_str += letter_mapping[int]
                else:
                    result_str += letter_mapping[int] + "|"
            result_str += "\n"
        result_str += "\n\n"
        
    # Test case
    if include_test:
        result_str += "Test case:\n"
        
        for line in task["test"][0]["input"]:
            for index, int in enumerate(line):
                if index == len(line) -1:
                    result_str += letter_mapping[int]
                else:
                    result_str += letter_mapping[int] + "|"
            result_str += "\n"

    return result_str


def find_task(task_identifier: str, *dicts: dict[str, Any]) -> dict[str, Any] | None:
    """
    Searches for task because it could be in either train or eval dict.

    Parameters:
        task_identifier (str): easy way to access the task via their string
        *dicts (dict[str, Any]): should always be train and eval dictionary

    Returns:
        part of the dict that the task is in or
        None if task was not found
    """
    for d in dicts:
        if task_identifier in d:
            return d[task_identifier]
    return None


def generate_expected_and_generated_outputs(task: dict[str, Any], generated_inputs: list) -> str:
    
    train_tasks = task['train']

    # Training Examples
    final_output = ""
    for i, task in enumerate(train_tasks):
        final_output += f"Example {i+1}: Expected Output\n["
        for index, row in enumerate(task["output"]):
            if index == len(task["output"])-1:
                final_output += f"\n{str(row)}"
            else:
                final_output += f"\n{str(row)},"
        final_output += f"]\n\nExample {i + 1}: Output generated by code\n[\n"
        
        if generated_inputs[i] is None:
            final_output += "No output generated.\n"
        else:
            for line_count,line in enumerate(generated_inputs[i]):
                if line_count == len(generated_inputs[i])-1:
                    final_output += str(line)
                else:
                    final_output += str(line) + ",\n"
        final_output += "]\n\n"
        
    return final_output

def format_outputs_for_prompt(generated_outputs: list) -> str:
    formatted_outputs = ""
    
    for i, output in enumerate(generated_outputs):
        formatted_outputs += f"Example {i+1} - Generated Output:\n[\n"
        if output is None:
            formatted_outputs += "No output generated.\n"
        else:
            for line_count, line in enumerate(output):
                if line_count == len(output)-1:
                    formatted_outputs += str(line)
                else:
                    formatted_outputs += str(line) + ",\n"
        formatted_outputs += "]\n\n"
    
    return formatted_outputs
        

def generate_code_fixing_prompt(task: dict[str, Any], extracted_answer: str, generated_inputs: list) -> list[dict]:
    
    system_prompt = """You are a very talented python programmer and puzzle solver. A colleague of yours was given a task but did not get the correct solution. He was given multiple paired example input and outputs. The outputs were produced by applying a transformation rule to the inputs. The task is to determine the transformation rule and implement it in code. The inputs and outputs are each 'grids'. A grid is a rectangular matrix of integers between 0 and 9 (inclusive). The integer values represent colors."""

    integer_grids = json_to_string(task, include_test=False)
    
    expected_and_generated_outputs = generate_expected_and_generated_outputs(task, generated_inputs)
    
    user_prompt = f"""Here are the examples of input and output grids represented as list[list[int]]:

{integer_grids}

The task is to reason about the transformation rule and then implement it in python code. The code needs to be in triple backticks (```python and then ```). Write one function named 'transform_grid()' which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You do not need to document code. You also do not need to test your code - it will be tested later.

This is the answer your colleague gave:

{extracted_answer}

Unfortunately, the code in this answer does not produce the exact correct results. Here are the the expected and produced outputs that the code produced:

{expected_and_generated_outputs}

Can you spot the problem? Please consider that a mistake can stem from the initial reasoning or the implementation in code. Carefully examine both the reasoning and the implementation.
Once you are done, please write correct code. Again, format your code within triple backticks (```python <your code> ```) and write one function that is named "transform_grid()".
You do not need to test the code, it will be tested for you later.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    return messages

def generate_continued_code_fixing_prompt(generated_inputs: list) -> str:
    
    formatted_outputs = format_outputs_for_prompt(generated_inputs)
    
    user_prompt = f"""It seems like the answer is still not quite correct. The generated outputs now look like this:
    
{formatted_outputs}
Please examine the output generated by the code you produced. Think about what went wrong. Consider that the problem may still either be in the initial hypothesis OR its implementation in code.
Reason carefully and for as long as necessary. Put that initial reasoning within <reasoning></reasoning> tags. Then correct the code accordingly. Make sure the corrected code you produce is still in a function named 'transform_grid()' and out all of the code within triple backticks (```python ```)."""

    return user_prompt

def generate_coding_prompt(task: dict[str, Any]) -> list[dict]:
    
    system_prompt = """You are a very talented python programmer. You will be given multiple paired example input and outputs. The outputs were produced by applying a transformation rule to the inputs. Your task is to determine the transformation rule and implement it in code. The inputs and outputs are each 'grids'. A grid is a rectangular matrix of integers between 0 and 9 (inclusive). The integer values represent colors. 
The transformation rule that you have to deduce might have multiple components and can be fairly complex. In your reasoning you will break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating your overall conclusion. This will avoid large leaps in reasoning. You reason in detail and for as long as is necessary to fully determine the transformation rule."""
    
    integer_grids = json_to_string(task, include_test=False)
        
    user_prompt = f"""Here are the examples of input and output grids represented as list[list[int]]:

{integer_grids}

You will need to carefully reason in order to determine the transformation rule. Put your reasoning in <reasoning></reasoning> tags. You break down complex problems into smaller parts and reason through them step by step, using sub-conclusions before coming to an overall conclusion. Large leaps in reasoning are not necessary. Take as long as you deem necessary.

- Determine the input and output grid sizes.
- Focus on what stays permanent and changes between input and output.
- Deduce a transformation rule and confirm that it works on the examples given.

If you find that you made a mistake in your previous reasoning, correct it. Make sure your reasoning does not only work on one but at least multiple of the given examples.

Once you are certain you found a valid transformation rule, implement it in python code. Put your code in triple backticks (```python and then ```). Write one function named 'transform_grid()' which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You do not need to document code. You also do not need to test your code - it will be tested later."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]
    
    return messages


def encode_image(image_path: str) -> str:
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
    
def generate_prompt_with_image(task_id: str, task_content: dict[str, Any]) -> list[dict]:
    # Find the image that belongs to the task and convert it into a format readable for the llm
    try:
        image_path = f"task_images/{task_id}.png"
        base64_image = encode_image(image_path)
    except:
        print("Problem finding or encoding the image for this task...")
    
    # Generate the text prompt
    integer_grids = json_to_string(task_content, include_test=True)
    
    system_prompt = """You are a very talented puzzle solver. You will be given multiple paired example input and outputs. The outputs were produced by applying a transformation rule to the inputs. Your task is to determine the transformation rule and solve the last puzzle (the 'test case'). The inputs and outputs are each 'grids'. A grid is a rectangular matrix of integers between 0 and 9 (inclusive). The integer values represent colors.
The transformation rule that you have to deduce might have multiple components and can be fairly complex. In your reasoning you will break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating your overall conclusion. This will avoid large leaps in reasoning. You reason in detail and for as long as is necessary to fully determine the transformation rule."""
    
    user_prompt = f"""Here are the examples of input and output grids. First you will be shown the grids as integer values. Each integer value maps to a color in the image as follows:
    black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.
    
Here are the grids as integers:

{integer_grids}

In addition, you are given an image of these grids. The image might not show all examples, but will give you an idea of what the tasks look like. In this image, the input grid will be on the left, the output grid on the right.

Now it is your turn. You will have to carefully reason in order to determine the transformation rule. Put your reasoning in <reasoning></reasoning> tags. You break down complex problems into smaller parts and reason through them step by step, using sub-conclusions before coming to an overall conclusion. Large leaps in reasoning are not necessary. Take as long as you deem necessary.

- First, look at the image provided to you. 
- Determine the input and output grid sizes.
- Determine what stays the same and what changes between input and output image.
- Proceed to do the same with the integer grids. Check if your observations from image and grid align.
- Deduce a transformation rule and confirm that it works on the examples given (both the image examples and the grid examples).

When you are done reasoning in detail, solve the test case and provide the correct output.
Please format your answer the same way, the examples were initially shown to you (as a list of list of integers). Please always format your answer like this:
Answer: ``` <your answer as a list of list of integers> ``` """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]
    
    # Add image representation if encoding was successful
    if base64_image:
        messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    else:
        print("Warning: prompt that was supposed to contain image was created without image representation.")

    return messages


def create_basic_prompt(task: dict[str, Any], allow_reasoning: bool = True) -> str | None:
    
    integer_grids = json_to_string(task, include_test=True)
    
    if allow_reasoning:
        task_instructions = """Now it is your turn. You will have to carefully reason in order to determine the transformation rule. Put your reasoning in <reasoning></reasoning> tags. You break down complex problems into smaller parts and reason through them step by step, using sub-conclusions before coming to an overall conclusion. Large leaps in reasoning are not necessary. Take as long as you deem necessary.

- Determine the input and output grid sizes.
- Look the integer representation to determine what stays the same and what changes between input and output.
- Deduce a transformation rule and confirm that it works on the examples given.

When you are done reasoning in detail, solve the test case and provide the correct output.
Please format your answer the same way, the examples were initially shown to you (as a list of list of integers). Please always format your answer like this:
Answer: ``` <your answer as a list of list of integers> ```"""

    else:
        task_instructions = """Now it is your turn to solve the test case given to you. Do NOT reason about the thought process.
Do NOT output an explanation of how you get to the answer. ONLY output the correct grid for the test case. Please format your answer the same way, the examples were initially shown to you (as a list of list of integers). Please always format your answer like this:
Answer: ``` <your answer as a list of list of integers> ```"""

    system_prompt = "You are a very talented puzzle solver. You will be given multiple paired example input and outputs. The outputs were produced by applying a transformation rule to the inputs. Your task is to determine the transformation rule and solve the last puzzle (the 'test case'). The inputs and outputs are each 'grids'. A grid is a rectangular matrix of integers between 0 and 9 (inclusive). The integer values represent colors."
    
    user_prompt = f"""Here are the examples of input and output grids:
    
{integer_grids}

{task_instructions}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    return messages
    


#################################
### WHY IS THIS IN HERE??? ######
# TODO: put this somewhere useful
def load_config(filepath="config.json"):
    """Loads the configuration from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_template_config(config, template_name=None):
    """Retrieves a specific template's configuration."""
    if template_name is None:
        template_name = config.get("default_template")

    templates = config.get("templates")
    if templates and template_name in templates:
        return templates[template_name]
    else:
        raise ValueError(f"Template '{template_name}' not found in config.")