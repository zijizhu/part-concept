
import os
import json
from openai import OpenAI
import pandas as pd
import pickle as pkl
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim



with open('CUB/gpt_res_cleaning.pkl', 'rb') as fp:
    gpt_res_cleaning = pkl.load(fp)

with open('CUB/gpt_res_generation.pkl', 'rb') as fp:
    gpt_res_generation = pkl.load(fp)

with open('CUB/gpt_res_gen_patch.pkl', 'rb') as fp:
    gpt_res_gen_patch = pkl.load(fp)



mappings = dict()
for part_res in gpt_res_cleaning:
    part_name, part_res_parsed = part_res['part_name'], json.loads(part_res['response'])
    to_remove, merge2dups = part_res_parsed['remove'], part_res_parsed['merge']
    merge = dict()
    for merge_to, duplicates in merge2dups.items():
        merge.update({d: merge_to for d in duplicates})
    mappings[part_name] = (to_remove, merge)


import itertools

def clean_concepts(concept_dict: dict):
    if all(type(cpt_list) is list for cpt_list in concept_dict.values()):
        return concept_dict
    cleaned_concept_dict = dict()
    # Here concepts is a dict of str or list
    for part_name, concepts in concept_dict.items():
        cleaned_concepts = []
        cleaned_concepts += itertools.chain(*[v if type(v) is list else [v]for v in concepts.values()])
        cleaned_concept_dict[part_name] = cleaned_concepts
    
    return cleaned_concept_dict

patch_dict = {patch_res['class_name']: patch_res['response'] for patch_res in gpt_res_gen_patch}

concepts_cleaned = dict()
for class_res in gpt_res_generation:
    class_name, class_concepts = class_res['class_name'], json.loads(class_res['response'])
    if class_name in patch_dict:
        class_concepts = json.loads(patch_dict[class_name])
    class_concepts = clean_concepts(class_concepts)
    class_cpts_processed = dict()
    for part_name, (removes, merges) in mappings.items():
        original_cpts = class_concepts[part_name]
        filtered_cpts = [cpt for cpt in original_cpts if cpt not in removes]
        mapped_concepts = [merges.get(cpt, cpt) for cpt in filtered_cpts]
        class_cpts_processed[part_name] = mapped_concepts
    # print(class_name)
    # print({k: len(v) for k, v in class_cpts_cleaned.items()})
    # Deal with responses that do not follow format
    counts = {k: len(v) for k, v in class_cpts_processed.items()}
    if any(v == 0 for v in counts.values()):
        print(class_name)
        print(counts)
        print(class_concepts)
    concepts_cleaned[class_name] = class_cpts_processed