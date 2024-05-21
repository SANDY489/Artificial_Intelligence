#%%
import json
import yaml
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
    sample = {"messages": [{"role": "system", "content": "You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number(0,1,2)."} , 
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
# Load API key
from openai import OpenAI
with open("api_keys.yaml", "r") as file:
    data = yaml.safe_load(file)
for api_key in data['api_keys']:
    if 'openai_gpt_key' in api_key:
        openai_gpt_key = api_key['openai_gpt_key']
client = OpenAI(api_key = openai_gpt_key)

#%% [markdown]
# Upload training file, use train2.jsonl later
client.files.create(
  file=open("train2.jsonl", "rb"),
  purpose="fine-tune"
)

#%% [markdown]
# Create a fine-tuned model
client.fine_tuning.jobs.create(
  training_file="file-XKH97TsZDuRY1bbEdRCVhvKP", 
  model="gpt-3.5-turbo-0125"
)


#%% [markdown]
# Load test.json
test_df = pd.read_json('dev.json')
test_df.head()

#%% [markdown]
# Create test list
test_list = []
for i in range(0, len(test_df)):
    # sample = {'sentence 1': item[0], 'sentence 2': item[1], 'classification': item[2]}
    sample = {"messages": [{"role": "system", "content": "You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number(0,1,2)."} , 
                        {"role": "user", "content": 'Sentence 1: '+str(test_df.loc[i][0])+"\n\n"+'Sentence 2: '+str(test_df.loc[i][1])}]}
    test_list.append(sample)
print(test_list)

#%% [markdown]
# Create test.jsonl
with open('dev.jsonl', 'w') as file:
    for data in test_list:
        json.dump(data, file)
        file.write('\n')  


#%%
with open('test.json') as f:
    data = json.load(f)
print(data)
#%% [markdown]
# Use a fine-tuned model
# This part not finished

from openai import OpenAI
client = OpenAI(api_key = openai_gpt_key)

for item in data:
    completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:iis-academia-sinica::9RD2EKa7",
    messages=[
        {"role": "system", "content": "You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number(0,1,2)."},
        {"role": "user", "content":  'Sentence 1: '+item[0]+"\n\n"+'Sentence 2: '+item[1]}
    ]
    )
    message_to_save = completion.choices[0].message
    print(message_to_save)


    

#%% [markdown]
# wait




