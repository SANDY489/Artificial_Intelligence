#%%
import json
import yaml
import openai 
import pandas as pd
import anthropic # for claude
from openai import OpenAI #update the latest openai version


# %% [markdown] 
# Read the training file
with open("train.json") as f:
    train_data = json.load(f)


# %% [markdown] 
# Create the specific training format for OpenAI
train_list = []
for item in train_data:
    # sample = {'sentence 1': item[0], 'sentence 2': item[1], 'classification': item[2]}
    sample = {"messages": [{"role": "system", "content": "You are a NLP researcher. This is an argument relation detection task. You should determine if there is a “Support” relation from sentence 1 to sentence 2 with output 1, else if it is an “Attack” relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number. "} , 
                        {"role": "user", "content": 'Sentence 1: '+item[0]+"\n\n"+'Sentence 2: '+item[1]}, 
                        {"role": "assistant", "content": item[2]}]}
    train_list.append(sample)

#%% [markdown]
# Create a New train.jsonl
with open("train.jsonl", 'w') as file:
    for data in train_list:
        json.dump(data, file)
        file.write('\n') 
#%% [markdown]
# Create a New train2.jsonl
with open("train2.jsonl", 'w', encoding='utf-8') as file:
    for data in train_list:
        json.dump(data, file, ensure_ascii=False)
        file.write(' n')

#%% [markdown]
# Upload training file, use train2.jsonl later
from openai import OpenAI
with open("api_keys.yaml", "r") as file:
    data = yaml.safe_load(file)
for api_key in data['api_keys']:
    if 'openai_gpt_key' in api_key:
        openai_gpt_key = api_key['openai_gpt_key']

client = OpenAI(api_key = openai_gpt_key)
client.files.create(
  file=open("train2.jsonl", "rb"),
  purpose="fine-tune"
)

#%% [markdown]
# Create a fine-tuned model
#train: FileObject(id='file-4pNFQae9Ew4Qtx7u8e77QYQx'
# train2: FileObject(id='file-d79KeGrEyG3jZwAwMnzuCEhg', bytes=4249567, created_at=1716231045, filename='train2.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
client.fine_tuning.jobs.create(
  training_file="file-d79KeGrEyG3jZwAwMnzuCEhg", 
  model="gpt-3.5-turbo"
)

#%% [markdown]
# Use a fine-tuned model
from openai import OpenAI
client = OpenAI(api_key = openai_gpt_key)

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
print(completion.choices[0].message)

#%% [markdown]
# wait
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


