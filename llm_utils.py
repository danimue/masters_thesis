import os
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List, Dict, Union

from mistralai import Mistral


def initialize_client(which: str = "mistral"):
    load_dotenv()  # Load variables from .env file
    
    if which == 'mistral':
        api_key = os.environ["MISTRAL_API_KEY"]
        client = Mistral(api_key=api_key)
        
    elif which == 'together_ai':
        client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )
        
    elif which == 'openrouter':
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
        )
    
    return client


def generate_llm_response(
    messages: list[dict],
    client=None,
    #model = "mistral-small-latest",
    #model: str = "qwen/qwen2.5-vl-72b-instruct:free",
    #model: str = "meta-llama/Llama-Vision-Free",
    model: str = None,
    base64_image: Optional[str] = None,
    extra_body: Optional[Dict] = None,
    num_samples: int = 1,
    temperature: float = 0.0,
    top_p = 1
    
) -> Optional[str]:
    """Generates a response from LLM. Image input optional."""

    all_responses = []
    start_time = time.time()
    
    try:
        completion = client.chat.completions.create(
        # mistral:
        #completion = client.chat.complete(
        
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=6000,
            n=num_samples
        )
        
        end_time = time.time()  # End timing
        processing_time = end_time - start_time
        print(f"Time taken for {num_samples} sample(s): {processing_time:.2f} seconds")
        
        additional_info = [round(processing_time, 2), num_samples, completion.usage.prompt_tokens, completion.usage.completion_tokens]
        
        if num_samples > 1:
            for choice in completion.choices:
                all_responses.append(choice.message.content)

            return all_responses, additional_info
        
        elif num_samples == 1:
            return completion.choices[0].message.content
        
        else:
            print("Invalid value for 'num_samples'.")
    
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None