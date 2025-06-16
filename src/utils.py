import pandas as pd 
import numpy as np

from config import cfg


def get_user_answer():
    answer = input()
    if answer == "y" or answer == "yes":
        return True
    return False


def create_file(path, append=False):
    from os.path import exists
    todo = "Append to" if append else "Overwrite" 
    if exists(path):
        print(f"The file {path} already exists. {todo} file? y/n: ")
        return get_user_answer()
    return True


def get_file_suffix(model):
    suffix = "_sub_iaa" if cfg["inter_annot"] else ""
    # if model == "gpt4":
    #     return f"_minimal_prompt{suffix}"
    return suffix


def parameter_check(conf_params, params={}):
    print("Make sure you set the following parameters in the config file correctly:")
    print("----------------------")
    for cparam in conf_params:
        print(cparam, ":", cfg[cparam])
    for param, value in params.items():
        print(param, ":", value)
    print("----------------------")
    print("Continue? y/n:")
    if not get_user_answer():
        exit()


'''
Returns a list of dataframes (one per argument)
df: the dataframe that should be split
col: the column where to split when empty
'''
def split_df_at_empty_lines(df, col="arg_id"):
    vals = [str(val) == "nan" for val in list(df[col])]
    subs = np.split(df, *np.where(vals))
    # remove potentially empty df at the end
    if len(subs[-1].index) == 1 and pd.isna(subs[-1].iloc[0,0]):
        subs = subs[:-1]
    return subs
    

'''
model: SentenceTransformer - either all-mpnet-base-v2 or ags21 (roberta large) model
'''
def get_embedding_overlap(model, sent1, sent2):
    from numpy.linalg import norm
    emb = model.encode([sent1, sent2])
    cos = np.dot(emb[0], emb[1])/(norm(emb[0])*norm(emb[1]))
    return round(cos, 3)
 

'''
Remove non-ASCII characters to get rid of wrong encodings
'''
def fix_encoding(mystring, replace_with="'"):
    # return "".join([c for c in mystring if ord(c) < 127])
    fixed = ""
    last_out_of_range = -3
    for i, c in enumerate(mystring):
        if ord(c) < 127:
            fixed += c
            continue
        elif i-1 != last_out_of_range:
            fixed += replace_with # "'"
        last_out_of_range = i
    return fixed

