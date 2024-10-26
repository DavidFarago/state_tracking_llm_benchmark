import random
from typing import Tuple, List

from openai import OpenAI 
import requests

MODEL_NAME = 'Qwen/Qwen2.5-72B-Instruct'
MODEL_SERVER = {
    'o1-mini': 'openai', 
    'o1-preview': 'openai', 
    'gpt-4o': 'openai',
    'llama-3.1-70b-versatile': 'groq', 
    'llama-3.2-11b-text-preview': 'groq', 
    'gemma2-9b-it': 'groq', 
    'lfm-40b': 'lambdalab',
    'Qwen/Qwen2.5-72B-Instruct': 'hyperbolic',
}
BASE_URL = {
    'openai': None, 
    'groq': 'https://api.groq.com/openai/v1', 
    'lambdalab': 'https://api.lambdalabs.com/v1',
    'hyperbolic': 'https://api.hyperbolic.xyz/v1/chat/completions'
    }
API_KEY = {
    'openai': 'see https://platform.openai.com/settings/profile?tab=api-keys', 
    'groq': 'see https://console.groq.com/keys', 
    'lambdalab':'see https://cloud.lambdalabs.com/api-keys',
    'hyperbolic': 'see https://app.hyperbolic.xyz/settings',
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
    if SERVER == 'hyperbolic':
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY[SERVER]}"
        }
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        response = requests.post(BASE_URL[SERVER], headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']
    output = SERVICE.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature
    )
    print(output)
    return output.choices[0].message.content

def check(model_name: str, swapList: List[Tuple[int, int]], final_order: List[int]) -> bool:
    prompt = """There are five slots:
* slot 1 contains ball 1,
* slot 2 contains ball 2,
* slot 3 contains ball 3,
* slot 4 contains ball 4,
* slot 5 contains ball 5.

""" 
    swap_instruction = ", ".join([f"swap ball {a} and ball {b}".format(a, b) for a, b in swapList])
    question = "Which ball is in which slot after all the swaps? Do not output anything but the final order of the balls, in the format <ball number in slot 1>, ..., <ball number in slot 5>, e.g. '1, 2, 3, 4, 5' if the balls happen to be in order."

    if MODEL_NAME[:2] == 'o1':
        output = llm_output(model_name, user_msg = f'{prompt}Now {swap_instruction}. {question}', temperature = 1.0)
    else:
        output = llm_output(model_name, sys_msg=prompt + question, user_msg= f'Swaps: {swap_instruction}')
    final_line = output.splitlines()[-1]
    print(final_line)
    print(final_order)
    return final_order == [int(x.strip()) for x in final_line.split(",")]

swapList, swapResultUpTo = generate_permutations()
length = 99
print(check(MODEL_NAME, swapList[:length+1], swapResultUpTo[length]))
