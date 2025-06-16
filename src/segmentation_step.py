import os
import pandas as pd
import nltk
import re

from config import cfg
from src.utils import create_file, parameter_check
import src.llms


def create_prompt(argument):
    approach = cfg["segmentation_approach"]
    with open(cfg["prompts"] + f"prompt_segmentation_{approach}.txt", "r") as file:
        prompt = file.read()
    with open(cfg["prompts"] + "example_text_school_uniforms.txt", "r") as file:
        example_text = file.read()
    with open(cfg["prompts"] + "example_segments_school_uniforms.txt", "r") as file:
        example_segments = file.read()
    
    prompt = prompt.replace("EEAA", example_text).replace("EEOO", example_segments)
    prompt = prompt.replace("TTAA", argument)
    return prompt


'''
Use some LLM to split texts
Returns a list of segments
'''
def split_with_llm(argument, topic):
    from src.llms import save_llm_output
    from src.utils import fix_encoding
    prompt = create_prompt(argument)
    prompting = getattr(src.llms, f"prompt_{cfg['segmentation_approach']}")
    response = prompting(prompt)
    save_llm_output(prompt, response, "segmentation", cfg["segmentation_approach"])
    response = response.replace("```", "")
    response = fix_encoding(response)
    no_args = ["does not contain any argumentative content", "no argumentative content", "does not provide arguments"]
    if any([phrase in response.lower() for phrase in no_args]):
        return [""]
    if response.startswith("[") and response.endswith("]"):
        response = response[1:-1]
    
    answer_split = response.split("\n")

    return [seg.strip() for seg in answer_split]


'''
Use TARGER API to split texts
Returns a list of segments
'''
def split_with_targer(argument):
    from targer_api import analyze_text
    claims, premises = [], []
    for sent in analyze_text(argument, model="tag-combined-fasttext"):
        claim, premise = "", ""
        for tok in sent:
            if tok.label.name == "C_B" or tok.label.name == "C_I":
                claim += tok.token + " "
            elif tok.label.name == "P_B" or tok.label.name == "P_I":
                premise += tok.token + " "
            else:
                if claim != "":
                    claims.append(claim.strip())
                    claim = ""
                if premise != "":
                    premises.append(premise.strip())
                    premise = ""

        if claim != "":
            claims.append(claim.strip())
        if premise != "":
            premises.append(premise.strip())
            
    return claims + premises


'''
Sentence segmentation approach
Returns a list of segments
'''
def split_by_sent(argument):
    sents = nltk.sent_tokenize(argument)
    return [sent for sent in sents if sent.count(" ") > 2]


'''
Paragraph segmentation with html-tags
Returns a list of segments
'''
def split_by_paragraph(argument, arg_id):
    from pathlib import Path
    from bs4 import BeautifulSoup
    if cfg["paragraphing"] == "html":
        breakat = re.compile(r'<br>\s?<br>|<p style="margin: [^>^"]*">')
        segments_clean = []
        for seg in re.split(breakat, argument):
            seg = seg.replace("</strong>", "").replace("</span>", "").replace("</em>", "").replace("</p>", "")
            text = BeautifulSoup(seg).get_text().replace("\n \n", " ").replace("\n", " ").strip()
            if text.count(" ") > 2:
                segments_clean.append(text)
    else:
        segments_clean = [arg.strip() for arg in argument.split("\n\n")]
    return segments_clean


'''
Choose segmentation approach
'''
def split_argument(argument, topic, arg_id):
    approach = cfg["segmentation_approach"]
    if approach in ["palm", "gpt4"]:
        return split_with_llm(argument, topic)
    elif approach == "targer":
        return split_with_targer(argument)
    elif approach == "sentence":
        return split_by_sent(argument)
    elif approach == "paragraph":
        return split_by_paragraph(argument, arg_id)
    else:
        print("This segmentation approach is not available: ", approach)
        exit()


'''
Iterate arguments from argsme and split them into segments
'''
def create_argument_segments(df, outname):
    for i, row in df.iterrows():
        arg = row["argument"]
        argument_segments = split_argument(arg, row["topic"], row["arg_id"])

        write_segments_to_file(argument_segments, row, row["topic"], outname, list(map(lambda x: "segment" if x == "argument" else x, list(df))))


'''
Write argument segments into csv file
cols should be: ['index', 'arg_id', 'argsme_id', 'segment', 'topic', 'stance', 'topic_in_argsme']
'''
def write_segments_to_file(argument_segments, info, topic, outname, cols):
    counter = 0
    data = []
    for seg in argument_segments:
        if seg == "" or " " not in seg:
            continue
        new_row = [info["index"], info["arg_id"] + "_" + str(counter),
                    info["argsme_id"], seg.strip(), topic, 
                    info["stance"], info["topic_in_argsme"]]
        data.append(new_row)
        counter += 1
    data.append([""] * len(cols))
    dfout = pd.DataFrame(data, columns=cols)
    dfout.to_csv(outname, mode="a", index=False, header=not os.path.exists(outname))


def run_segmentation():
    from src.utils import get_file_suffix
    approach = cfg["segmentation_approach"]
    indata = cfg["argument_texts"]
    print("segment", indata)

    dataset = cfg["dataset"] if cfg["dataset"] == "" else "_"+cfg["dataset"]
    outname = cfg["split_data"] + f"{approach}/segments{dataset}_{approach}{get_file_suffix(approach)}.csv"
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    encoding = "ISO-8859-1"
    
    if "paragraph" in approach:
        indata = cfg["argument_texts_html"]
    
    if cfg["check_params"]:
        parameter_check([], {"approach": approach, "argument_texts": indata, "outname": outname})

    if create_file(outname, append=True):
        df = pd.read_csv(indata, encoding=encoding)
        create_argument_segments(df, outname)
    
    return outname


if __name__ == "__main__":
    segments_file = run_segmentation()
