import random
import time
from typing import Tuple, List

from openai import OpenAI 

COT_INTENSITY = 1 # 0: no CoT, 1: standard CoT
FORBID_COMMON_ERRORS = False # this addition to the prompt seldomly improves the performance of the models
MODEL_NAME = 'lfm-40b'
MODEL_SERVER = {
    'o1-mini': 'openai', 
    'o1-preview': 'openai', 
    'gpt-4o': 'openai',
    'gpt-4o-mini': 'openai',
    'llama-3.1-70b-versatile': 'groq', 
    'llama-3.2-11b-text-preview': 'groq', 
    'gemma2-9b-it': 'groq', 
    'lfm-40b': 'lambdalab',
    'Qwen/Qwen2.5-72B-Instruct': 'hyperbolic',
    'jamba-1.5-large': 'ai21',
}
BASE_URL = {
    'openai': None, 
    'groq': 'https://api.groq.com/openai/v1', 
    'lambdalab': 'https://api.lambdalabs.com/v1',
    'hyperbolic': 'https://api.hyperbolic.xyz/v1/chat/completions',
    'ai21': None,
    }
API_KEY = {
    'openai': 'see https://platform.openai.com/settings/profile?tab=api-keys', 
    'groq': 'see https://console.groq.com/keys', 
    'lambdalab': 'see https://cloud.lambdalabs.com/api-keys',
    'hyperbolic': 'see https://app.hyperbolic.xyz/settings',
    'ai21': 'see https://studio.ai21.com/account/api-key',
    }
SERVER = MODEL_SERVER[MODEL_NAME]
SERVICE = OpenAI(api_key=API_KEY[SERVER], base_url=BASE_URL[SERVER])

def generate_permutations() -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    random.seed(42)
    swapList = []
    swapResultUpTo = []
    order = [1, 2, 3, 4, 5]

    for _ in range(1000):
        a, b = random.sample([1, 2, 3, 4, 5], 2)
        swapList.append((a, b))
        i = order.index(a)
        j = order.index(b)
        order[i], order[j] = order[j], order[i]
        swapResultUpTo.append(order.copy())

    return swapList, swapResultUpTo

def llm_output(model_name: str, sys_msg: str = '', user_msg: str = '', temperature: float = 0.0) -> str:
    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    if user_msg:
        messages.append({"role": "user", "content": user_msg})
    print(messages)

    start_time = time.perf_counter()
    if SERVER == 'hyperbolic':
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY[SERVER]}"
        }
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        response = requests.post(BASE_URL[SERVER], headers=headers, json=data).json()
        result = response['choices'][0]['message']['content']
    elif SERVER == 'ai21':
        from ai21 import AI21Client
        from ai21.models.chat import ResponseFormat, ChatMessage
        client = AI21Client(api_key=API_KEY[SERVER])
        proprietary_messages = [ChatMessage(content=m['content'], role=m['role']) for m in messages]
        response = client.chat.completions.create(
            model=model_name,
            messages=proprietary_messages,
            temperature=temperature,
            response_format=ResponseFormat(type="text"),
        )
        result = response.choices[0].message.content
    else:
        response = SERVICE.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature
        )
        result = response.choices[0].message.content
    end_time = time.perf_counter()
    print(f"Inference time: {end_time - start_time:.6f} seconds")
    try:
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Reasoning tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
    except:
        pass
    return result

def check(model_name: str, swapList: List[Tuple[int, int]], final_order: List[int]) -> bool:
    prompt = """There are five slots:
* slot 1 contains ball 1,
* slot 2 contains ball 2,
* slot 3 contains ball 3,
* slot 4 contains ball 4,
* slot 5 contains ball 5.

""" 
    swap_instruction = ", ".join([f"swap ball {a} and ball {b}".format(a, b) for a, b in swapList])
    questions = [
        "Which ball is in which slot after all the swaps? Do not output anything but the final order of the balls, in the format <ball number in slot 1>, ..., <ball number in slot 5>, e.g. '1, 2, 3, 4, 5' if the balls happen to be in order.", 
        "Which ball is in which slot after all the swaps? Output the order after each swap in the format <ball number in slot 1>, ..., <ball number in slot 5>, e.g. '1, 2, 3, 4, 5' if the balls happen to be in order.",
    ]
    question = questions[COT_INTENSITY]
    if FORBID_COMMON_ERRORS:
        question += " Never miss any given swap. Never mix up balls and slots."

    if MODEL_NAME[:2] == 'o1':
        output = llm_output(model_name, user_msg = f'{prompt}Now {swap_instruction}. {question}', temperature = 1.0)
    else:
        output = llm_output(model_name, sys_msg=prompt + question, user_msg= f'Swaps: {swap_instruction}')
    print(output if COT_INTENSITY > 0 else output.splitlines()[-1])
    print(final_order)
    return final_order == [int(x.strip()) for x in output.splitlines()[-1].split(",")]

swapList, swapResultUpTo = generate_permutations()
length = 73
print(check(MODEL_NAME, swapList[:length+1], swapResultUpTo[length]))
