from sentence_transformers import SentenceTransformer

config_default = {
    "check_params": True,

    "segmentation_approach": "sentence",    # palm, gpt4, sentence, paragraph, ajjour, targer
    "paragraphing": "html",                 # how to split into paragraphs: html, default (i.e., \n\n)
    
    # files
    "argument_texts": "data/split_data/arguments_txt.csv",
    "argument_texts_html": "data/split_data/arguments_html.csv",
    "key_statements_groundtruth": "data/split_data/manual/key_statements_ground_truth.csv",
    
    "inter_annot": False,
    "dataset": "",                          # empty for original args.me texts

    # segment matching
    "manual_matching_file": "data/split_data/segments_palm_manual_matching_to_key_statements.csv",
    "matching_approaches": ["ngram", "sequencematcher", "transformer"],
    "n_ngram": 3,
    "thresh_ngram": 0.3,    # 0.3
    "thresh_seq": 0.5,      # 0.5
    "thresh_transf": 0.9,   # 0.8

    # filter non-argumentative segments
    "filter_column": "arg_class",
    "filter": False,

    # general
    "sent_transformer": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),

    # general paths
    "argsme_data": "data/argsme_data/",
    "data_inter_annot_segmentation": "data/split_data/segmentation_inter_ann/",
    "ibm_data": "data/ibm_data/",
    "prompts": "data/prompts/",
    "split_data": "data/split_data/",
    "annotator_data": "data/split_data/segmentation_inter_ann/",
    "argsme_v1-0_csv": "args-me-1.0-cleaned-meta.csv",
    "argsme_csv": "argsme_2020-04-01_filtered.csv",
    "ags21_similarity_model": "models/final-experiment-on-training-data/roberta-large-final-model-fold-0-2023-07-05_15-10-46",
}

cfg = config_default
def set_config(config_name):
    global cfg
    for key, val in eval(config_name).items():
        cfg[key] = val