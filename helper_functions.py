import numpy as np
from typing import List, Tuple, Dict, Optional

####
####    Determine Grid Sizes
####
def get_grid_sizes(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], Optional[Tuple[int, int]]]:
    """
    Extracts the dimensions (height, width) of the input and output grids
    for each training example and the input grid of the test example in an ARC task.

    Args:
        task: A dictionary representing an ARC challenge task.
              It should have a 'train' key containing a list of examples,
              where each example is a dictionary with 'input' and 'output' keys,
              both holding 2D lists (grids) of integers.
              It may also have a 'test' key containing a list with exactly one
              example having an 'input' key with a 2D list (grid).

    Returns:
        A tuple containing:
        - A list of tuples. Each tuple corresponds to a training example and
          contains two tuples: the shape (height, width) of the input grid
          and the shape (height, width) of the output grid.
        - An optional tuple representing the shape (height, width) of the
          input grid of the test example, or None if no test example or input
          grid is found.
    """
    train_grid_sizes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    train_examples = task.get("train", [])

    for example in train_examples:
        input_grid = example.get("input")
        output_grid = example.get("output")

        input_shape = (len(input_grid), len(input_grid[0]) if input_grid and input_grid[0] else 0)
        output_shape = (len(output_grid), len(output_grid[0]) if output_grid and output_grid[0] else 0)

        train_grid_sizes.append((input_shape, output_shape))

    test_input_size: Optional[Tuple[int, int]] = None
    test_examples = task.get("test")
    if test_examples and len(test_examples) == 1:
        test_input_grid = test_examples[0].get("input")
        if test_input_grid:
            test_input_size = (len(test_input_grid), len(test_input_grid[0]) if test_input_grid[0] else 0)

    return train_grid_sizes, test_input_size

def format_grid_sizes_to_string(train_grid_sizes: List[Tuple[Tuple[int, int], Tuple[int, int]]], test_input_size: Optional[Tuple[int, int]]) -> Optional[str]:
    """
    Formats the list of training input and output grid sizes and the test input
    grid size into a human-readable string (more concise).
    Condenses the output if all training input or all training output grid sizes are the same.
    Ensures training input size information appears before output size information for each case.
    Includes the input size of the test example.
    """
    if not train_grid_sizes and test_input_size is None:
        return "No grid sizes provided."

    parts = ["Grid sizes:\n"]

    # Training grid sizes
    if train_grid_sizes:
        first_in, first_out = train_grid_sizes[0]
        all_in_same = all(in_size == first_in for in_size, _ in train_grid_sizes)
        all_out_same = all(out_size == first_out for _, out_size in train_grid_sizes)

        # Input size information for training examples
        if all_in_same:
            parts.append(f"All training input grid sizes are {first_in[0]}x{first_in[1]}")
        else:
            parts.extend([f"Training Example {i + 1} Input: {in_size[0]}x{in_size[1]}"
                          for i, (in_size, _) in enumerate(train_grid_sizes)])

        # Output size information for training examples
        if all_out_same:
            parts.append(f"All training output grid sizes are {first_out[0]}x{first_out[1]}")
        else:
            parts.extend([f"Training Example {i + 1} Output: {out_size[0]}x{out_size[1]}"
                           for i, (_, out_size) in enumerate(train_grid_sizes)])

    # Test input grid size
    if test_input_size:
        parts.append(f"Test Input Grid Size: {test_input_size[0]}x{test_input_size[1]}")
    else:
        parts.append("\nTest Input: No test input grid found.")

    return "\n".join(parts)

###
### Find Symmetries
###
def check_grid_symmetries(grid: List[List[int]]) -> List[str]:
    """
    Checks for various symmetries in a given grid.

    Args:
        grid: A 2D list (list of lists) representing the grid of integers.

    Returns:
        A list of strings, where each string describes a symmetry found in the grid.
        The list will be empty if no symmetries are detected.
    """
    np_grid = np.array(grid)
    symmetries = []

    # Horizontal Symmetry
    if np.array_equal(np_grid, np.flipud(np_grid)):
        symmetries.append("Horizontal Symmetry")

    # Vertical Symmetry
    if np.array_equal(np_grid, np.fliplr(np_grid)):
        symmetries.append("Vertical Symmetry")

    # Rotational Symmetry (180 degrees)
    if np.array_equal(np_grid, np.rot90(np_grid, k=2)):
        symmetries.append("180-degree Rotational Symmetry")

    # Rotational Symmetry (90 degrees clockwise)
    if np.array_equal(np_grid, np.rot90(np_grid, k=1)):
        symmetries.append("90-degree Clockwise Rotational Symmetry")

    # Diagonal Symmetry (top-left to bottom-right)
    if np_grid.shape[0] == np_grid.shape[1] and np.array_equal(np_grid, np_grid.T):
        symmetries.append("Diagonal Symmetry (Top-Left to Bottom-Right)")

    # Anti-Diagonal Symmetry (top-right to bottom-left)
    if np_grid.shape[0] == np_grid.shape[1] and np.array_equal(np_grid, np.fliplr(np_grid).T):
        symmetries.append("Anti-Diagonal Symmetry (Top-Right to Bottom-Left)")

    return symmetries

def check_task_symmetries(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Tuple[Dict[int, Dict[str, List[str]]], Optional[List[str]]]:
    """
    Checks for symmetries in the input and output grids of all training examples
    and the input grid of the test example within an ARC task.

    Args:
        task: A dictionary representing an ARC challenge task.
              It should have a 'train' key containing a list of examples,
              where each example is a dictionary with 'input' and 'output' keys,
              both holding 2D lists (grids) of integers.
              It may also have a 'test' key containing a list with exactly one
              example having an 'input' key with a 2D list (grid).

    Returns:
        A tuple containing:
        - A dictionary where keys are the indices of the training examples (starting from 0).
          Each value is another dictionary with two keys: 'input' and 'output',
          with lists of strings describing the symmetries.
        - An optional list of strings describing the symmetries found in the input
          grid of the test example, or None if no test example or input grid is found.
    """
    all_train_symmetries: Dict[int, Dict[str, List[str]]] = {}
    train_examples = task.get("train", [])

    for i, example in enumerate(train_examples):
        input_grid = example.get("input")
        output_grid = example.get("output")

        input_symmetries = check_grid_symmetries(input_grid) if input_grid else []
        output_symmetries = check_grid_symmetries(output_grid) if output_grid else []

        all_train_symmetries[i] = {
            "input": input_symmetries,
            "output": output_symmetries,
        }

    test_input_symmetries: Optional[List[str]] = None
    test_examples = task.get("test")
    if test_examples and len(test_examples) == 1:
        test_input_grid = test_examples[0].get("input")
        if test_input_grid:
            test_input_symmetries = check_grid_symmetries(test_input_grid)

    return all_train_symmetries, test_input_symmetries

def format_task_symmetries_to_string(all_train_symmetries: Dict[int, Dict[str, List[str]]], test_input_symmetries: Optional[List[str]]) -> str:
    """
    Formats the symmetry information for all training examples and the test input
    into a string.

    Args:
        all_train_symmetries: A dictionary where keys are the indices of training examples
                              and values are dictionaries containing 'input' and 'output'
                              keys, each with a list of found symmetries.
        test_input_symmetries: An optional list of strings describing the symmetries
                               found in the input grid of the test example.

    Returns:
        A string describing the symmetries found in the input and output grids
        of each training example, and the input grid of the test example.
    """
    output_string = "Symmetry Information:\n"

    found_any_train_symmetry = False
    for symmetries in all_train_symmetries.values():
        if symmetries.get('input') or symmetries.get('output'):
            found_any_train_symmetry = True
            break

    if found_any_train_symmetry:
        for example_index, symmetries in all_train_symmetries.items():
            output_string += f"Example {example_index + 1}:\n"

            input_syms = symmetries.get('input', [])
            if input_syms:
                output_string += f"  Input Grid: {', '.join(input_syms)}\n"
            else:
                output_string += "  Input Grid: No symmetries found.\n"

            output_syms = symmetries.get('output', [])
            if output_syms:
                output_string += f"  Output Grid: {', '.join(output_syms)}\n"
            else:
                output_string += "  Output Grid: No symmetries found.\n"
            output_string += "\n"
    else:
        output_string += "No symmetries were found in any of the training examples.\n\n"

    if test_input_symmetries is not None:
        if test_input_symmetries:
            output_string += f"Test Input Grid: {', '.join(test_input_symmetries)}\n"
        else:
            output_string += "Test Input Grid: No symmetries found."
    else:
        output_string += "Test Input Grid: Not available.\n"

    return output_string.strip()

###
### Create Grid that shows pixel level changes
###
def get_change_grid_text(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> str: # Changed return type hint to str
    """
    Creates a text-based representation of the changes between input and output grids
    for each training example. Only generates output if *each* input-output pair
    has the exact same shape. Cells are separated by "|".

    Args:
        task: A dictionary representing an ARC challenge task.

    Returns:
        A string containing the change grids for all training examples where each
        example's input/output shapes matched, preceded by an introductory sentence.
        Returns an empty string if the task has no training examples or if *any*
        training example has mismatched input/output shapes.
    """
    train_examples = task.get("train") # Use .get for safety
    if not train_examples:
        # Return empty string if no training examples exist
        return "" # Original code returned "", sticking to that

    # --- MODIFIED SECTION START ---
    # Check if *each* input/output pair has matching dimensions.
    # If any pair does not match, return an empty string immediately.
    all_pairs_match_internally = True
    for i, example in enumerate(train_examples):
        input_grid = example.get('input')
        output_grid = example.get('output')

        # Handle cases where input or output might be missing or empty
        if input_grid is None or output_grid is None:
             print(f"Warning: Example {i} missing input or output grid. Skipping size check for this pair.")
             # Depending on requirements, you might want to fail here:
             # all_pairs_match_internally = False
             # break
             continue # Or continue checking others

        # Calculate shapes robustly, handling potentially empty grids
        in_rows = len(input_grid)
        in_cols = len(input_grid[0]) if in_rows > 0 else 0
        out_rows = len(output_grid)
        out_cols = len(output_grid[0]) if out_rows > 0 else 0

        if in_rows != out_rows or in_cols != out_cols:
            # If *any* pair has mismatched input/output shapes, set flag and break
            all_pairs_match_internally = False
            break

    # If the check failed for any pair, return empty string
    if not all_pairs_match_internally:
        return ""
    # --- MODIFIED SECTION END ---

    # If we reach here, all pairs passed the internal shape consistency check.
    # Proceed to build the output string.

    output_string = "Additionally, you are given the following grids that show the changes between input and output. If an example pair stays the same in a certain cell, that will be marked with ' -- '. A change from input to output will be shown by 'value_input' -> 'value_output'.\n\n" # Added newline for spacing
    for i, example in enumerate(train_examples):
        # We re-fetch grids here, safe because we passed the check loop
        input_grid = example['input']
        output_grid = example['output']

        # We know shapes match from the check above
        shape_rows = len(input_grid)
        shape_cols = len(input_grid[0]) if shape_rows > 0 else 0

        # Add Example title only if there are rows/cols to show
        if shape_rows > 0 and shape_cols > 0:
            output_string += f"Example {i + 1} Changes ({shape_rows}x{shape_cols}):\n" # Added shape info
            change_grid_rows = []

            for r in range(shape_rows):
                row_changes = []
                for c in range(shape_cols):
                    if input_grid[r][c] == output_grid[r][c]:
                        # Ensure consistent spacing, pad if necessary
                        row_changes.append(" -- ")
                    else:
                        # Pad single-digit numbers for better alignment (optional)
                        in_val_str = str(input_grid[r][c])
                        out_val_str = str(output_grid[r][c])
                        # Simple padding example:
                        change_str = f"{in_val_str.rjust(1)}->{out_val_str.ljust(1)}"
                        # Adjust rjust/ljust width if dealing with >9 values or prefer different alignment
                        row_changes.append(change_str)
                change_grid_rows.append("|".join(row_changes)) # Removed map(str, ...) as elements are already strings

            output_string += "\n".join(change_grid_rows) + "\n\n" # Add extra newline between examples
        else:
             # Optionally note if an example had empty grids even if shapes matched (0x0)
             output_string += f"Example {i + 1} Changes: (Empty Grid)\n\n"


    return output_string.strip() # Remove leading/trailing whitespace only at the very end