import pandas as pd
from nltk import sent_tokenize
from pathlib import Path
from difflib import SequenceMatcher, Differ

from src.utils import get_file_suffix
from config import cfg


def get_exact_mappings(df_text, df_preds):
    data = []
    foundf, foundff, not_found = 0, 0, 0
    for i, row in df_text.iterrows():
        arg_id = row["arg_id"]
        text = row["argument"]
        df_segs = df_preds[df_preds["arg_id"] == arg_id]
        start = 0
        for j, seg_row in df_segs.iterrows():
            seg = str(seg_row["segment"]).lower()
            if seg == "nan":
                continue
            sents = sent_tokenize(seg)
            # simplest case: segment unmodified
            # if multiple sentences in segment: find sentences separately
            if all([sent.lower() in text.lower() for sent in sents]):
                if len(sents) == 1:
                    foundf += 1
                else:
                    foundff += 1
                extracted, start, end = find_exact_mapping_boundaries(text, sents, start)
                data.append([arg_id, extracted, start, end])
                start = end
            else:
                data.append([arg_id, "XXX", "", ""])
                not_found += 1
        data.append(["", "", "", ""])
    print(f"found: {foundf}, {foundff}, not found: {not_found}")
    return data, not_found


def find_exact_mapping_boundaries(text, sents, substart):
    '''
    If (single sentences of) a segment can be mapped exactly, return contiguous passage
    containing (all sentences of) the segment
    '''
    for i, sent in enumerate(sents):
        sub, substart, subend = extract_seg(sent.lower(), text.lower(), substart)
        if i == 0:
            start = substart
    end = subend
    return text[start:end].strip(), start, end



def map_segments_to_original_text(model):
    '''
    Main method to match (potantially modified) segments to the original text
    - param model: the segmentation model
    '''
    print(model)
    df_text = pd.read_csv(cfg["argument_texts"], encoding="ISO-8859-1")
    outpath = Path(cfg["split_data"], f"matched_to_text/{model}_extractions_2.csv")
    suffix = get_file_suffix(model)
    matchfile = f"{cfg['split_data']}{model}/sample_split_train_segments_{model}{suffix}.csv"
    df_preds = pd.read_csv(matchfile)
    df_preds["arg_id"] = df_preds["arg_id"].str.lower()
    if str(df_preds["arg_id"][0]).count("_") == 3:
        df_preds["arg_id"] = [str(id).rsplit("_", 1)[0] for id in list(df_preds["arg_id"])]
    
    df_preds = df_preds.drop([i for i,row in df_preds.iterrows() 
                              if str(row["segment"]) != "nan" and str(row["segment"]).count(" ") <= 2])
    print("num preds", len(df_preds.index))
    
    # exact mappings of complete segments
    print("exact mappings")
    data, not_found = get_exact_mappings(df_text, df_preds)
    dfout = pd.DataFrame(data, columns=["arg_id", "segment", "start", "end"])
    dfout.to_csv(outpath, index=False)

    # half and similar segments
    if not_found > 0: # "XXX" in list(dfout["segment"]):
        dfout = pd.read_csv(outpath)
        print("half and similar segments") 
        updated_data, not_found = post_iterate(dfout, df_preds, df_text)
        dfout = pd.DataFrame(updated_data, columns=["arg_id", "segment", "start", "end"])
        dfout.to_csv(outpath.replace("_2", "_3"), index=False)

    # fill gaps between identified segments
    if not_found > 0: # "XXX" in list(dfout["segment"]): 
        dfout = pd.read_csv(outpath.replace("_2", "_3"))
        print("fill gaps between identified segments")
        updated_data, not_found = post_iterate(dfout, df_preds, df_text, fill_gaps=True)
        dfout = pd.DataFrame(updated_data, columns=["arg_id", "segment", "start", "end"])
        dfout.to_csv(outpath.replace("_2", "_4"), index=False)
    
    # most similar passages based on sequencematcher
    if not_found > 0: # "XXX" in list(dfout["segment"]): 
        dfout = pd.read_csv(outpath.replace("_2", "_4"))
        print("find most similar passages with sequencematcher")
        updated_data = get_sequence_matches(dfout, df_preds, df_text)
        dfout = pd.DataFrame(updated_data, columns=["arg_id", "argument", "start", "end"])
        dfout.to_csv(outpath.replace("_2", "_5"), index=False)


def get_sequence_matches(extracted_data, df_preds, df_text):
    
    model = cfg["sent_transformer"]
    data = []
    found, not_found = 0, 0
    start = 0
    seq_thresh = 0.35
    prev_arg_id = ""
    # search for "XXX", i.e., not matched segments
    for i, row in extracted_data.iterrows():
        if str(row["segment"]) != "XXX":
            data.append(list(row))
            continue
        arg_id = row["arg_id"]
        if prev_arg_id != arg_id:
            prev_arg_id = arg_id
            start = 0
        text = list(df_text.loc[df_text["arg_id"] == arg_id]["argument"])[0]
        text_sents = sent_tokenize(text[start:])    # sentences of original arg. text

        seg_row = df_preds.iloc[[i]]
        seg = list(seg_row["segment"])[0]
        seg_sents = sent_tokenize(seg)
        indexes, max_values = [], []
        # iterate sentences in predicted seg.
        prev_index = -1
        for s, sent in enumerate(seg_sents):
            similarities = [round(SequenceMatcher(None, sent, t_sent).ratio(), 3) for t_sent in text_sents]
            if len(similarities) == 0:
                continue

            max_sim = max(similarities)
            max_index = similarities.index(max_sim)
            if max_sim <= seq_thresh:
                break
            # if the most similar original sentence is more than 20 sentences distant from the
            # previous most similar sentence: choose closer (and less similar) sentence
            while max_index - prev_index > 20 and prev_index != -1:
                max_sim = max(similarities)
                if max_sim > seq_thresh:
                    max_index = similarities.index(max_sim)
                    similarities[max_index] = 0
                else:
                    break
            indexes.append(max_index)
            max_values.append(max_sim)
            prev_index = max_index

        if len(indexes) > 0:
            extracted, start, end = find_exact_mapping_boundaries(
                text, [text_sents[min(indexes)], text_sents[max(indexes)]], start)
            data.append([arg_id, extracted, start, end])
            start = end
            found += 1
        else:
            data.append([arg_id, "XXX", "", ""])
            not_found += 1

    print(f"found: {found}, not found: {not_found}")
    return data


def fill_gap_row(start, end, text, argid_before, argid_after):
    '''
    For not found segments surrounded by found segments, map segment to passage between neighboring boundaries
    '''

    # start = extracted_data.iloc[[i-1]].end.iloc[0]    # end of seg before
    # end = extracted_data.iloc[[i+1]].start.iloc[0]    # start of seg after
    if str(start) == "nan" and argid_before == "nan":   # seg at text beginning
        start = 0
    if str(end) == "nan" and argid_after == "nan":      # seg at text end
        end = len(text)
    if str(start) == "nan" or str(end) == "nan":
        return None, None, None
    else:
        return text[int(start):int(end)].strip(), start, end


def get_start(data, j, arg_id):
    id = arg_id
    while id == arg_id:
        j -= 1
        row = data[j]
        id = str(row[0])
        if str(row[3]) != "nan" and str(row[3]) != "":
            return int(row[3])
    return 0
    

def get_end(extracted_data, j, arg_id):
    id = arg_id
    while id == arg_id:
        j += 1
        row = extracted_data.iloc[[j]]
        id = str(row["arg_id"])
        if str(row.start.iloc[0]) != "nan":
            return int(row.start.iloc[0])
    return -1


def post_iterate(extracted_data, df_preds, df_text, fill_gaps=False):
    '''
    Search for non-exact mappings
    param fill_gaps=False -> search for mappings of half segments, or for similar passages
    param fill_gaps=True  -> if neighboring segments are identified, apply their boundaries to not found segment
    param extracted_data: dataframe with exactly mapped segments or XXX if not found
    param df_preds: dataframe of predicted segments that should be mapped to original text
    param df_text: original argument texts
    '''

    data = []
    found, foundhalf, foundsim, filled, not_found = 0, 0, 0, 0, 0
    for i, row in extracted_data.iterrows():
        arg_id = row["arg_id"]
        extracted = row["segment"]
        if str(arg_id) != "nan" and (str(extracted) == "nan" or str(extracted) == "XXX"):
            text = list(df_text.loc[df_text["arg_id"] == arg_id]["argument"])[0]
            argid_before = str(extracted_data.iloc[[i-1]].arg_id.iloc[0])
            argid_after = str(extracted_data.iloc[[i+1]].arg_id.iloc[0])
            # for single segment: map complete text     # TODO or try to find half first?
            if fill_gaps:
                fseg, start, end = fill_gap_row(extracted_data.iloc[[i-1]].end.iloc[0], extracted_data.iloc[[i+1]].start.iloc[0],
                                       text, argid_before, argid_after)
                if fseg == None:
                    data.append(list(row))
                    not_found += 1
                else:
                    data.append([arg_id, fseg, start, end])
                    filled += 1
                continue

            if argid_before == "nan" and argid_after == "nan":
                data.append([arg_id, text, 0, len(text)])
                continue
            
            start = get_start(data, i, arg_id)
            end = get_end(extracted_data, i, arg_id)
            end = len(text) if end == -1 else end
            seg = df_preds.iloc[[i]].argument.iloc[0]
            
            if isinstance(seg, float):
                print("***", i, seg)
                data.append([arg_id, "XXX", "", ""])
                not_found += 1
                continue
            # if complete segment not found: find half segment
            extracted, start, end = find_half(start, end, seg, text)
            if extracted != None:
                foundhalf += 1
                data.append([arg_id, extracted.strip(), start, end])
                start = end
            else:
                # if complete and half segment not found: find similar passage
                extracted, start, end = find_similar_sequence(start, seg, text)
                if extracted != None:
                    foundsim += 1
                    data.append([arg_id, extracted.strip(), start, end])
                else:
                    data.append([arg_id, "XXX", "", ""])
                    not_found += 1
                start = end

        # segment already mapped
        else:
            data.append(list(row))

    if fill_gaps:
        print(f"filled: {filled}, not found: {not_found}")
    else:
        print(f"found: {found}, {foundhalf}, {foundsim}, not found: {not_found}")
    return data, not_found


def find_similar_sequence(start, seg, text):
    '''
    If no exact match is found for complete segment or half of the segment,
    find similar sequence with difflib
    '''
    
    miss, add, match = 0, 0, 0
    diff = Differ().compare(seg, text[start:])
    non_add = 0
    for i, line in enumerate(list(diff)[non_add:]):
        if line.startswith("+"):
            add += 1
        elif line.startswith("-"):
            miss += 1
        elif line.startswith("?"):
            print("not found")
            return None, start, start
        else:
            match += 1
        
        if i >= len(seg) and start+i <= len(text) and match >= len(seg)/4*3:
            end = find_clean_boundary(
                text, start+i, max(start, start+i-int(len(seg)/4)), min(len(text), start+i+int(len(seg)/4)))
            return text[start:end], start, end
        
    return None, start, start


def find_clean_boundary(text, current, minb, maxb):
    '''
    For segments where boundary is estimated: try to cut at reasonable punctuation marks
    - search before and behind initial cutoff and cut at closer punctuation mark (with bias in favor of behind cutoffs, i.e., find_next)
    param current: initial cutoff
    param minb: minimal boundary (if snippet==half2: end of previous segment, if snippet==half1: end of found segment)
    param maxb: maximal boundary (if snippet==half2: beginning of found segment, if snippet==half1: TODO )
    '''

    next = find_next(text, current, maxb)
    prev = find_previous(text, minb, current)
    return prev if 2*(current-prev) < (next-current) else next


def find_next(text, start, end):
    for i in range(start, end):
        if text[i] in [".", ";", "?"]:
            return i+1
    for i in range(start, end):
        if text[i] == " ":
            return i
    return start


def find_previous(text, start, end):
    for i in reversed(range(start, end)):
        if text[i] in [".", "-", ")"]:
            return i+1
    return start


def extract_seg(seg, text, idx, snippet="complete"):
    '''
    Extract a segment given a start index
    (either for complete mapping, or if only half of the segment could be mapped exactly)
    param seg: the segment to extract
    param text: text in which to find segment
    param idx: end of previous segment (or 0 for first segment)
    param seg_len: length of segment to extract
    param text_before: length of 'removed' half of segment, if only second half of a segment is matched
    '''

    seg_start = text.lower().find(seg, idx)
    seg_end = seg_start + len(seg)
    if snippet == "half1": # if first half of segment is mapped: find end boundary of un-mapped second half
        seg_end = find_clean_boundary(text, min(len(text), seg_start+(2*len(seg))), seg_start+len(seg), len(text))
    elif snippet == "half2": # if second half of segment is mapped: find start boundary of un-mapped first half
        if seg_start-len(seg) < idx:
            seg_start = idx
        else:
            seg_start = find_clean_boundary(text, max(idx, seg_start-len(seg)), idx, seg_start)
    return text[seg_start:seg_end].strip(), seg_start, seg_end


def find_half(start, end, seg, text):
    '''
    Try to map half of a given segment to the original text
    '''
    half = int(len(seg)/2)
    half1 = seg[:half]
    if half1 in text.lower():
        return extract_seg(half1, text, start, "half1")
    
    half2 = seg[half:]
    if half2 in text.lower():
        return extract_seg(half2, text, start, "half2")
    
    return None, start, end


def adapt_seg_boundaries(model):
    '''Get correct segment boundaries after manual revision'''
    from src.utils import fix_encoding
    filepath = Path(cfg["split_data"], f"matched_to_text/{model}_extractions_manual_revision.csv")
    df = pd.read_csv(filepath, encoding="iso-8859-1")
    ref_df = pd.read_csv(cfg["argument_texts"], encoding="iso-8859-1")
    last_start = 0
    for i, row in df.iterrows():
        seg0 = str(row["segment"])
        seg = fix_encoding(seg0)
        if seg != "nan":
            text = list(ref_df[ref_df["arg_id"] == row["arg_id"]]["argument"])[0]
            text = fix_encoding(text)
            seg_start = text.find(seg) if text.count(seg)<2 else text.find(seg, last_start)
            if text.count(seg) > 1:
                print(f"* {i}", seg)
            seg_end = seg_start + len(seg)
            if seg_start == -1:
                print(f"- seg not found at i={i} -> check boundaries")
            else:
                last_start = seg_start
            df.at[i, "start"] = seg_start
            df.at[i, "end"] = seg_end
    df.to_csv(filepath.replace(".csv", "_corr.csv"), index=False)


