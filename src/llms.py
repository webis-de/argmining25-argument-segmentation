import requests
from openai import OpenAI
from pathlib import Path

from src.credentials import openai_api_key, palm_url
from config import cfg



def prompt_gpt4(prompt):
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="gpt-4o-mini") #"gpt-4-1106-preview"
    return response.choices[0].message.content #.replace("\n\n", " ")


def prompt_palm(prompt="Test prompt", temperature=0.5, max_output_tokens=1200, top_p=1.0, top_k=40):
    data = {"prompt": prompt, "temperature": temperature, "max_output_tokens": max_output_tokens, "top_p": top_p, "top_k": top_k}
    response = requests.post(palm_url, json=data)
    if response.status_code == 200:
        result = response.json()
        return(result["output"])
    else:
        print("Error:", response.status_code, "\n", response.text)



def save_llm_output(prompt, response, task, model):
    import os
    import datetime
    import pandas as pd
    time = str(datetime.datetime.now()).split(" ")[0]
    df = pd.DataFrame([[prompt, response, time]], columns=["prompt", "response", "time"])

    folder = Path(cfg["split_data"], "llm_responses/")
    filepath = Path(folder, f"{task}_{model}.csv")
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    df.to_csv(filepath, mode="a", index=False, header=not os.path.exists(filepath))


if __name__ == "__main__":
    prompt = "Say hello."
    prompt_gpt4(prompt)
