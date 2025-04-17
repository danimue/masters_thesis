import os
import requests
import zipfile
import shutil
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from io import BytesIO
from PIL import Image
import numpy as np

from typing import Tuple, Optional


from typing import Dict, List, Any

def check_dataset() -> None:
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


def filter_tasks_by_grid_size(tasks: Dict[str, Dict[str, List[Dict[str, List[List[int]]]]]]) -> Dict[str, Dict[str, List[Dict[str, List[List[int]]]]]]:
    """
    Filters a dictionary of ARC training tasks to include only those where
    all input and output grids in all examples (under the 'train' key) have
    a total size (rows * cols) less than or equal to 225.

    Args:
        train_tasks: A dictionary where keys are task IDs and values are
                     dictionaries containing 'train' (and potentially 'test')
                     keys, with lists of training/testing examples.

    Returns:
        A new dictionary containing only the tasks that meet the grid size criteria.
    """
    filtered_tasks: Dict[str, Dict[str, List[Dict[str, List[List[int]]]]]] = {}
    for task_id, task_data in tasks.items():
        train_examples = task_data.get('train', [])
        if not train_examples:
            continue  # Skip tasks without training examples

        include_task = True
        for example in train_examples:
            input_grid = example.get('input')
            output_grid = example.get('output')

            input_rows = len(input_grid)
            input_cols = len(input_grid[0]) if input_rows > 0 else 0
            output_rows = len(output_grid)
            output_cols = len(output_grid[0]) if output_rows > 0 else 0

            # if (input_rows > 15 or input_cols > 15 or input_rows * input_cols > 225 or
            #         output_rows > 15 or output_cols > 15 or output_rows * output_cols > 225):
            #     include_task = False
            if (input_rows * input_cols > 225 or output_rows * output_cols > 225):
                include_task = False
                break  # No need to check other examples in this task

        if include_task:
            filtered_tasks[task_id] = task_data
    return filtered_tasks














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



def visualize_task(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Image.Image:
    
    # Color mapping
    cmap = colors.ListedColormap([
    'black', 'blue', 'red', 'green', 'yellow', 
    'grey', 'pink', 'orange', 'purple', 'brown'
    ])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    # Determine the number of rows and columns for the subplot grid
    num_examples = len(task['train'])
    num_rows = num_examples
    num_cols = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    for i in range(num_examples):
        mat_inp = np.array(task['train'][i]['input'])
        shape_i = mat_inp.shape
        mat_out = np.array(task['train'][i]['output'])
        shape_o = mat_out.shape

        # Input subplot
        ax_inp = axes[i, 0]  # Access the correct subplot
        ax_inp.imshow(mat_inp, cmap=cmap, norm=norm)
        ax_inp.set_title(f"Example {i+1} Input - Shape: {shape_i}")
        ax_inp.set_xticks(np.arange(-.5, shape_i[1], 1))
        ax_inp.set_yticks(np.arange(-.5, shape_i[0], 1))
        ax_inp.grid(color='gray', linestyle='-', linewidth=1)
        ax_inp.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        # Output subplot
        ax_out = axes[i, 1]  # Access the correct subplot
        ax_out.imshow(mat_out, cmap=cmap, norm=norm)
        ax_out.set_title(f"Example {i+1} Output - Shape: {shape_o}")
        ax_out.set_xticks(np.arange(-.5, shape_o[1], 1))
        ax_out.set_yticks(np.arange(-.5, shape_o[0], 1))
        ax_out.grid(color='gray', linestyle='-', linewidth=1)
        ax_out.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    # Save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Close matplotlib figure to free up memory
    
    image = Image.open(buf)    
    return image


def plot_arc_grid(ax, grid_data: np.ndarray, title: str, cmap, norm, grid_color='#555555'):
    """Helper function to plot a single ARC grid onto a Matplotlib Axes."""
    shape = grid_data.shape
    ax.imshow(grid_data, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    # Use minor ticks aligned to the edges of pixels for grid lines
    ax.set_xticks(np.arange(-.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, shape[0], 1), minor=True)
    # Grid lines based on minor ticks
    ax.grid(which='minor', color=grid_color, linestyle='-', linewidth=1)
    # Hide major tick labels and marks
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    # Set border color/width
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
        spine.set_linewidth(1)

# --- Main Visualization Function (Simplified) ---
def visualize_task_simplified(
    task: Dict[str, List[Dict[str, List[List[int]]]]]
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Generates simplified visualizations for training and test examples of an ARC task.

    Args:
        task: The ARC task dictionary.

    Returns:
        A tuple (train_image, test_image), where each element is a PIL Image
        or None if data is missing.
    """
    train_image = None
    test_image = None

    # Shared color mapping and normalization
    cmap = colors.ListedColormap([
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = colors.Normalize(vmin=0, vmax=9)

    # --- Generate Training Image ---
    if task.get('train'): # Check if 'train' key exists and list is not empty
        num_examples = len(task['train'])
        fig_train, axes_train = plt.subplots(
            num_examples, 2,
            figsize=(6, 3 * num_examples), squeeze=False
        )
        fig_train.suptitle("Training Examples", fontsize=14)

        for i, pair in enumerate(task['train']):
            if 'input' not in pair or 'output' not in pair:
                print(f"Warning: Skipping train example {i} due to missing input/output.")
                axes_train[i, 0].set_title(f"Input {i+1} (Missing)")
                axes_train[i, 1].set_title(f"Output {i+1} (Missing)")
                axes_train[i, 0].axis('off')
                axes_train[i, 1].axis('off')
                continue

            mat_inp = np.array(pair['input'])
            mat_out = np.array(pair['output'])

            # Use helper function to plot
            plot_arc_grid(axes_train[i, 0], mat_inp, f"Input {i+1} ({mat_inp.shape[0]}x{mat_inp.shape[1]})", cmap, norm)
            plot_arc_grid(axes_train[i, 1], mat_out, f"Output {i+1} ({mat_out.shape[0]}x{mat_out.shape[1]})", cmap, norm)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        buf_train = BytesIO()
        fig_train.savefig(buf_train, format='png', bbox_inches='tight', dpi=200)
        buf_train.seek(0)
        train_image = Image.open(buf_train)
        plt.close(fig_train)
    else:
        print("No valid 'train' data found.")

    # --- Generate Test Image ---
    if task.get('test'): # Check if 'test' key exists and list is not empty
        test_pair = task['test'][0] # Assume only one test pair
        if 'input' in test_pair and 'output' in test_pair:
            mat_inp_test = np.array(test_pair['input'])
            mat_out_test = np.array(test_pair['output'])

            fig_test, axes_test = plt.subplots(1, 2, figsize=(7, 4), squeeze=False)
            fig_test.suptitle("Test Example", fontsize=14)

            # Use helper function to plot
            plot_arc_grid(axes_test[0, 0], mat_inp_test, f"Test Input ({mat_inp_test.shape[0]}x{mat_inp_test.shape[1]})", cmap, norm)
            plot_arc_grid(axes_test[0, 1], mat_out_test, f"Test Output ({mat_out_test.shape[0]}x{mat_out_test.shape[1]})", cmap, norm)

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            buf_test = BytesIO()
            fig_test.savefig(buf_test, format='png', bbox_inches='tight', dpi=200)
            buf_test.seek(0)
            test_image = Image.open(buf_test)
            plt.close(fig_test)
        else:
             print("Test example missing 'input' or 'output' key.")
    else:
        print("No valid 'test' data found.")

    return train_image, test_image