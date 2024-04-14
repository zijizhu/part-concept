import os
import json
import argparse
import pickle as pkl
from openai import OpenAI
from collections import defaultdict


def chat_completion_cub(client, input_str: str, example_input_str: str, example_output_str: str, part_name: str):
    messages = [
        {"role": "system",
        "content": ("You are a helpful bird assistant who helps with filtering visual features "
                    "that are useful to identify different species of bird from photos. You will output JSON.")},
        # One shot example    
        {"role": "user",
        "content": (f"Given a set of candidate visual concepts about head of different species of bird, "
                    "1. Merge concepts if they are too similar in terms of textual similarity. "
                    "2. Remove a concept if it is not a visual feature of bird's head that can be recognized from photos. "
                    "The remaining concepts should be concise and distinct. Output the merged and removed concepts in JSON format.")},

        {"role": "user", "content": f"Input: {example_input_str}"},
        {"role": "user", "content": "Output:"},
        {"role": 'assistant', "content": example_output_str},
        
        {"role": "user",
        "content": (f"Given a set of candidate visual concepts about {part_name}s of different species of bird, "
                    "1. Merge concepts if they are too similar in terms of textual similarity. "
                    f"2. Remove a concept if it is not a visual feature of bird's {part_name} that can be recognized from photos. "
                    "The remaining concepts should be concise and distinct. Output the merged and removed concepts in JSON format.")},

        {"role": "user", "content": f"Input: {input_str}"},
        {"role": "user", "content": "Output:"}
    ]
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_format={ "type": "json_object" },
        messages=messages
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generating concepts usig GPT4.')
    parser.add_argument('--gpt_response_path', type=str)
    parser.add_argument('--example_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    client = OpenAI()

    with open(args.example_path, 'r') as fp:
        example = json.load(fp=fp)    

    # Read and sort part-wise concepts generated bny GPT
    with open(args.gpt_response_path, 'rb') as fp:
        gpt4_res = pkl.load(fp)

    concept_set = defaultdict(set)  # dict of part_name: set_of_concepts
    for res in gpt4_res:
        class_name, response_str = res['class_name'], res['response']
        response_dict = json.loads(response_str)
        for k, v in response_dict.items():
            concept_set[k].update(v)

    concept_set_sorted = dict()  # dict of part_name: list_of_sorted_concepts
    for k, v in concept_set.items():
        concept_set_sorted[k] = sorted(list(v))

    # Read in example
    example_input_str = '; '.join(example['input'])
    example_output_str = json.dumps(example['output'])

    all_responses = []
    for part_name, concept_list in concept_set_sorted.items():
        input_str = '; '.join(concept_list)
        response = chat_completion_cub(client, input_str, example_input_str, example_output_str, part_name)
        all_responses.append(dict(part_name=part_name, response=response))
        with open(args.output_path, 'wb') as fp:
            pkl.dump(all_responses, file=fp)
 
