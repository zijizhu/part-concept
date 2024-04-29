import os
import json
import argparse
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from openai import OpenAI


def chat_completion_cars(client, class_name: str, example_class_name: str, example_answer: str, example_template: str):
    messages = [
        {"role": "system",
        "content": "You are a helpful assistant designed to provide visual features of different models of cars and output them in JSON."},

        {"role": "system",
        "content": f"Please provie only visual features that can be identified from images or photos. Output in the following JSON format: {example_template}"},
        
        {"role": "user",
        "content": (f"To visually identify {example_class_name}, please provide 3 most distinguishable characteristics for each of {example_class_name}'s parts "
                    "(front, side, back) in shape and color: ")},
        
        {"role": 'assistant', "content": example_answer},
        
        {"role": "user",
        "content": (f"To visually identify {class_name}, please provide 3 most distinguishable characteristics of each of {class_name}'s parts "
                    "(front, side, back) in shape, color, or size:")}
    ]
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_format={ "type": "json_object" },
        messages=messages
    )
    return response.choices[0].message.content


def chat_completion_cub(client, class_name: str, example_class_name: str, example_answer: str, example_template: str):
    messages = [
        {"role": "system",
        "content": "You are a helpful assistant designed to provide visual features of different species of birds and output them in JSON."},

        {"role": "system",
        "content": f"Output in the following JSON format: {example_template}"},
        
        {"role": "user",
        "content": (f"To visually identify {example_class_name}, please provide 3 most common characteristics of each of {example_class_name}'s parts "
                    "(head, beak, tail, wing, eye, leg, torso) in shape, color, or size:")},
        
        {"role": 'assistant', "content": example_answer},
        
        {"role": "user",
        "content": (f"To visually identify {class_name}, please provide 3 most common characteristics of each of {class_name}'s parts "
                    "(head, beak, tail, wing, eye, leg, torso) in shape, color, or size:")}
    ]
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_format={ "type": "json_object" },
        messages=messages
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generating concepts usig GPT4.')
    parser.add_argument('--class_names_path', type=str, help='Path to a file with class names.')
    parser.add_argument('--dataset', type=str, choices=['CUB', 'CARS'])
    parser.add_argument('--example_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()
    client = OpenAI()

    with open(args.example_path, 'r') as fp:
        example = json.load(fp=fp)

    example_class_name = example['class']
    example_concepts_str = json.dumps(example['concepts'])
    example_template_str = json.dumps(example['template'])

    # class_df= pd.read_csv(args.class_names_path, sep=' ', header=None, names=['class_id', 'class_name'])
    # class_names = class_df['class_name'].str.replace('_', ' ').str.split('.').str[-1].to_list()
    with open(args.class_names_path, 'r') as fp:
        class_names = fp.read().splitlines()
    print('class names:')
    print(class_names)

    if args.dataset == 'CUB':
        chat_completion_fn = chat_completion_cub
    elif args.dataset == 'CARS':
        chat_completion_fn = chat_completion_cars
    else: raise NotImplemented

    all_responses = []
    class_names_pbar = tqdm(class_names)
    for name in class_names_pbar:
        class_names_pbar.set_postfix_str(name)
        response = chat_completion_fn(client, name, example_class_name, example_concepts_str, example_template_str)
        all_responses.append(dict(class_name=name, response=response))
        with open(args.output_path, 'wb') as fp:
            pkl.dump(all_responses, file=fp)
 
