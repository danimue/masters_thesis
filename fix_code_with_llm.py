from utils import check_dataset, load_files_from_json, find_task, create_basic_prompt, generate_coding_prompt, generate_prompt_with_image
from llm_utils import generate_llm_response, initialize_client
from eval import extract_solution, evaluate_solution, fix_output_string, extract_code, execute_transform_function

import time
import json

import numpy as np
import pandas as pd

check_dataset()
train, eval = load_files_from_json()

config = {
        #'model': 'meta-llama/Llama-Vision-Free',
        #'model': 'pixtral-large-latest',
        # 'model': 'qwen/qwen2.5-vl-72b-instruct:free',
        #'model': 'google/gemma-3-27b-it:free',
        'model': 'codestral-latest',
        'solution_type': 'simple',
        'temperatures': [0.0],
        'samples_per_step': 1,
    }

df = pd.read_csv("results/program_synthesis/codestral_full.csv")