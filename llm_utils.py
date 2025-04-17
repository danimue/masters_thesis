import os
import time
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List, Dict, Union, Generator
from openai import AsyncOpenAI
from mistralai import Mistral

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]


def initialize_client(which: str = "mistral"):
    load_dotenv()  # Load variables from .env file
    
    if which == 'mistral':
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        
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
        
    elif which == 'openrouter_async':
        client = AsyncOpenAI(
        # If the environment variable is not configured, replace the following line with Bailian API Key: api_key="sk-xxx",
        api_key=os.environ["OPENROUTER_API_KEY"], 
        base_url="https://openrouter.ai/api/v1"
        )
        
    elif which == 'alibaba':
        client = OpenAI(
        # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
        api_key=os.environ["DASHSCOPE_API_KEY"], 
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        
    elif which == 'alibaba_async':
        client = AsyncOpenAI(
        # If the environment variable is not configured, replace the following line with Bailian API Key: api_key="sk-xxx",
        api_key=os.environ["DASHSCOPE_API_KEY"], 
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
    
    return client



async def task(question):
    print(f"Sending question: {question}")
    response = await client.chat.completions.create(
        messages=[
            {"role": "user", "content": question}
        ],
        model="qwen-plus", # Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
    )
    print(f"Received answer: {response.choices[0].message.content}")
    
    
async def generate_llm_response_async(
    messages: list[dict],
    client=None,
    model: str = None,
    temperature: float = 0.0,
    top_p = 1) -> str | None:
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=8192,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return None


def query_reasoning_model(messages):

    url = "https://openrouter.ai/api/v1/chat/completions"

    payload = {
        "model": 'qwen/qwq-32b',
        "messages": messages,
        "reasoning": {
          "exclude": False},
        "top_p": 0.7,
        "max_tokens": 64000,
        "temperature": 0.6,
      }

    headers = {
        "Authorization": f"Bearer {os.environ.get("OPENROUTER_API_KEY")}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        answer = response.json()

        output = answer['choices'][0].get('message', {}).get('content')
        reasoning = answer['choices'][0].get('message', {}).get('reasoning')

        return output, reasoning

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing response: {e}")
        return None, None

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
        #completion = client.chat.completions.create(
        # mistral:
        completion = client.chat.complete(
        
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=8192,
            n=num_samples
        )
        
        end_time = time.time()  # End timing
        processing_time = end_time - start_time
        print(f"Time taken for {num_samples} sample(s): {processing_time:.2f} seconds")
        
        additional_info = [round(processing_time, 2), num_samples, completion.usage.prompt_tokens, completion.usage.completion_tokens]
        
        if num_samples > 1:
            print("This many choices were returned: " + str(len(completion.choices)))
            for choice in completion.choices:
                all_responses.append(choice.message.content)

            return all_responses
        
        elif num_samples == 1:
            return completion.choices[0].message.content
        
        else:
            print("Invalid value for 'num_samples'.")
    
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None
    
    
#######################################################
# Specifically for Qwen-QwQ Streaming in Demo-Version #
#######################################################
def stream_response_qwen_qwq(
    prompt_messages: List[Dict[str, str]]
) -> Generator[str, None, None]:
    """
    Calls the LLM API with streaming and yields text chunks (deltas) as they arrive.
    Raises exceptions on API or critical stream processing errors.

    Args:
        prompt_messages: The list of messages forming the prompt.

    Yields:
        str: Text chunks (content deltas) received from the stream.

    Raises:
        ValueError: If API key is not set.
        requests.exceptions.RequestException: If the API request fails (e.g., connection error, timeout, bad status code).
        RuntimeError: For critical errors during stream processing.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("ERROR: OPENROUTER_API_KEY not set.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen/qwq-32b",
        "messages": prompt_messages,
        "top_p": 0.7,
        "temperature": 0.6,
        "max_tokens": 64000,
        "stream": True,
        "reasoning": {"exclude": False}
    }

    try:
        buffer = ""
        req_timeout = 300  # 5 minutes timeout
        # Use a context manager for the request
        with requests.post(
            url, headers=headers, json=payload, stream=True, timeout=req_timeout
        ) as response:
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                buffer += chunk
                while "\n" in buffer:
                    line_end = buffer.find("\n")
                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1 :]

                    if line.startswith("data: "):
                        data_str = line[len("data: ") :]
                        if data_str == "[DONE]":
                            # Normal stream end, just finish the generator
                            return # Exit the generator cleanly

                        try:
                            data_obj = json.loads(data_str)
                            delta = data_obj.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content # YIELD THE TEXT DELTA

                        except json.JSONDecodeError:
                            print(f"\nWarning: Non-JSON data received: {data_str}") # Log warning
                            pass # Ignore and continue processing stream
                        except Exception as e_json:
                            # Raise a specific error for issues processing valid SSE lines
                            raise RuntimeError(f"Error processing stream chunk JSON: {e_json} - Data: '{data_str}'") from e_json

            # If loop finishes without returning (e.g., stream closed unexpectedly)
            print("\nWarning: Stream ended without explicit [DONE] message.")

    except requests.exceptions.RequestException as e_req:
        # Raise request exceptions directly for caller to handle
        print(f"\nERROR: API Request Failed: {e_req}")
        raise e_req
    except Exception as e_runtime:
        # Catch other unexpected errors during setup or streaming
        print(f"\nERROR: Stream Processing Failed Unexpectedly: {e_runtime}")
        # Re-raise or raise a custom error
        raise RuntimeError(f"Stream processing failed: {e_runtime}") from e_runtime