import os
import json
import requests
import zipfile
import shutil

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
        for row in test_task[0]["input"]:
            final_output += f"\n{str(row)}"
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


def create_prompt(task: dict[str, Any], prompt_type: str = 'simple', reason: bool = True) -> str | None:
    #prompt = ""
    
    if prompt_type == 'simple':
        task_explanation = """The examples below show input-output pairs of numerical representations of visual grids. Each integer value represents a colored square on this grid. There are 10 possible colors (and therefore integer values) - 0 to 9, each representing a color. Each example input grid is transformed into an output grid. Each integer value represents a colored square on that grid. 
The input is transformed into the corresponding output using the same transformation rule every time. Your task is to analyse this transformation rule by observing the examples and then solve the test-case.\n\n"""

        task_examples = json_to_string(task, include_test=True)
        
        grid_representation_char = json_to_ascii(task, include_test=True)
        task_examples_char = f"""\n\nHere is another representation of the grids given to you. These examples are the same as in the list[list[int]] representation.
They are given to you as an additional help to understand the problem. This representation shows "black" / integer 0 parts as empty. This is because black is usually
the background color of the grid and it is supposed to make it easier to detect objects in the grid.

{grid_representation_char}"""
        
        if reason:        
            task_instructions = """\n\nNow it is your turn. Please reason about what the transformation rule could be. Put your thoughts between <reasoning> ... </reasoning> tags. Then, solve the test case and provide the correct output.
Please always format your answer like this, as correct formatting is crucial to this task:
Answer: ``` <your answer> ```"""
        else:
            task_instructions = """\n\nNow it is your turn to solve the test case given to you. Do NOT reason about the thought process.
Do NOT output an explanation of how you get to the answer. ONLY output the correct grid for the test case. Format your answer like this:
Answer: ``` <your answer> ``` Make sure to format your answer as a list of lists of integers, just like the examples shown to you."""

        return task_explanation + task_examples + task_instructions

    elif prompt_type == 'program_synthesis':
        task_explanation = """Here are are some examples of the transformation from `input` to `output`: \n"""
        task_examples = json_to_string(task, include_test=False)
        task_instructions = """\n\nYou'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

For your reasoning, please pay close attention to where the objects in the grid are in the input and where they are moved in the output example. You can consider a space with a 0 as an empty space, as 0 represents the color black.

Once you have finished your first round of reasoning, please start a second round of reasoning in <reasoning></reasoning> tags. In this round, consider each row of the input and output array. According to your transformation logic, would they be the same? If not, what went wrong and how can it be fixed?
If you made any mistakes, correct them. If you are unsure that you have found the correct transformation rule, you may start as many rounds of <reasoning></reasoning> as you deem necessary.

After your reasoning is completed write code in triple backticks (```python and then ```). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation which works in general (it shouldn't just work for the additional input).

Don't write tests in your python code, just output the `transform` function (it will be tested later)."""

        return task_explanation + task_examples + task_instructions

    else:
        print("No valid prompt type specified...")
        return None
    
    
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