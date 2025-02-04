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
    
    print("Current working directory:", os.getcwd())
    print("Script location:", os.path.abspath(__file__))

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
                print(file)
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
def json_to_string(task: dict) -> str:
    """
    Converts a given ARC-task from its json-like form to a string, so that it can be used in a prompt.
    
    
    Parameters:
        - task (str): string identifier for each task
    """

    #TO-DO: Make it so that you can reference tasks by string identifier or loop over train/eval dict?
    
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

    # Test Case
    # omit test case for now
    final_output += "Test\n["
    for row in test_task[0]["input"]:
        final_output += f"\n{str(row)}"
    final_output += "]"

    return final_output


def find_task(task_identifier: str, *dicts: Dict[str, Any]) -> Any:
    """_summary_

    Args:
        task_identifier (str): _description_
        dicts (Dict[str, Any]): _description_
    Returns:
        Any: _description_
    """
    for d in dicts:
        if task_identifier in d:
            return d[task_identifier]
    return None


def create_prompt(task_identifier: str) -> str:
    
    task = find_task(task_identifier, train, eval)
    if task != None:
        stringified_grid = json_to_string(task)
    
    task_explanation = """Here is are some examples of the transformation from `input` to `output`: \n
"""

    # task_instructions = """\n\nPlease return the missing output grid with the same formatting as the examples. Do not write code, do not explain your solution - just answer with the correct grid."""
    
    task_examples = stringified_grid
    
    task_instructions = """You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

For your reasoning, please pay close attention to where the objects in the grid are in the input and where they are moved in the output example. You can consider a space with a 0 as an empty space, as 0 represents the color black.

Once you have finished your first round of reasoning, please start a second round of reasoning in <reasoning></reasoning> tags. In this round, consider each row of the input and output array. According to your transformation logic, would they be the same? If not, what went wrong and how can it be fixed?
If you made any mistakes, correct them. If you are unsure that you have found the correct transformation rule, you may start as many rounds of <reasoning></reasoning> as you deem necessary.

After your reasoning is completed write code in triple backticks (```python and then ```). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation which works in general (it shouldn't just work for the additional input).

Don't write tests in your python code, just output the `transform` function (it will be tested later)."""

    return task_explanation + task_examples + task_instructions


check_dataset()
train, eval = load_files_from_json()
prompt = create_prompt("0d3d703e")
print(prompt)