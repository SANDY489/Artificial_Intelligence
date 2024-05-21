#%%
import json
import yaml
import openai #update the latest openai version
import pandas as pd
import anthropic # for claude
from openai import OpenAI



#%%
def load_data(file):
    with open(file) as f:
        data = json.load(f)
    return data

def create_train_list(data):
    """
    {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, 
                   {"role": "user", "content": "What's the capital of France?"}, 
                   {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
    train_list=[]
for i in range(0, len(df)):
    sample={"messages": [{"role": "system", "content": "You are a NLP researcher. This is am argument mining task. You should determine the given two sentence is support each other with output 1, attack each other with output 2, none with output 0. Only output one numeric number. "} , 
                         {"role": "user", "content": 'Argument1:'+str(df.loc[i][1])+"\n\n"+'Argument2:'+str(df.loc[i][2])}, 
                         {"role": "assistant", "content": "0"}]}
 
    train_list.append(sample)

                   """
    train_list = []
    for item in data:
        # sample = {'sentence 1': item[0], 'sentence 2': item[1], 'classification': item[2]}
        sample = {"messages": [{"role": "system", "content": "You are a NLP researcher. This is an argument relation detection task. You should determine if there is a “Support” relation from sentence 1 to sentence 2 with output 1, else if it is an “Attack” relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number. "} , 
                         {"role": "user", "content": 'Sentence 1: '+item[0]+"\n\n"+'Sentence 2: '+item[1]}, 
                         {"role": "assistant", "content": item[2]}]}
        
    train_list.append(sample)
    return train_list

def load_config(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    for api_key in data['api_keys']:
        if 'openai_gpt_key' in api_key:
            openai_gpt_key = api_key['openai_gpt_key']
        elif 'claude3_key' in api_key:
            claude3_key = api_key['claude3_key']
    return openai_gpt_key, claude3_key



if __name__ == "__main__":

    #load api keys
    openai_key, claude3_key = load_config("api_keys.yaml")
    # print(openai_key, claude3_key)

    #load train, dev file
    train_data = load_data('train.json')
    train_list = create_train_list(train_data)
    # print(train_list)





    eval_data = load_data('dev.json')
    
    #load openai
    # openai_api(openai_key)


