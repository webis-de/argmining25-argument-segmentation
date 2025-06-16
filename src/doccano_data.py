import json
import pandas as pd


def extract_doccano_segments(inpath, outpath):
    '''
    Extract continuous segments from original text with start and end index
    given a jsonl file with segment annotations from doccano
    '''
    
    data = []   # arg_id, segment, start, end
    with open(inpath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        arg_id = result["arg_id"]
        if "entities" in result.keys():     # e.g. "entities":[{"id":82,"label":"S1","start_offset":0,"end_offset":79}]
            dictd = True
            labels_sorted = sorted(result["entities"], key=lambda d:d["start_offset"]) 
        else:                               # e.g. "label":[[0,79,"S1"]]
            dictd = False
            labels_sorted = result["label"]
        if len(labels_sorted) == 0:
            data.append([arg_id, "", "", ""])
        seen_labels = ["nan"]
        for seg in labels_sorted:
            start = seg["start_offset"] if dictd else seg[0]
            end = seg["end_offset"] if dictd else seg[1]
            label = seg["label"] if dictd else seg[2]
            # if a segment is not labeled continuously: include unlabeled part inbetween
            if label == seen_labels[-1]:
                start = data[-1][-2]
                del data[-1]
            seen_labels.append(label)
            extracted = result["text"][start:end]
            data.append([arg_id, extracted, start, end])
        data.append(["", "", "", ""])
    
    dfout = pd.DataFrame(data, columns=["arg_id", "argument", "start", "end"])
    dfout.to_csv(outpath, index=False)

