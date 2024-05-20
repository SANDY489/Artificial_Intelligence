# %%

import json
import openai #update the latset openai version
import pandas as pd

# Load the JSON file into a dataframe
df = pd.read_json('file path')

# Display the dataframe
df.head()

# %%
path = 'api txt file path'
with open(path) as f:
    for line in f.readlines():
        my_key=line
openai.api_key = my_key

# %%


# %%
train_list=[]
for i in range(0, len(df)):
    sample={"messages": [{"role": "system", "content": "You are a NLP researcher. This is am argument mining task. You should determine the given two sentence is support each other with output 1, attack each other with output 2, none with output 0. Only output one numeric number. "} , 
                         {"role": "user", "content": 'Argument1:'+str(df.loc[i][1])+"\n\n"+'Argument2:'+str(df.loc[i][2])}, 
                         {"role": "assistant", "content": "0"}]}
 
    train_list.append(sample)

# %%
train = "train.jsonl"

# Open the file in write mode and write each dictionary as a JSON object on a separate line
with open(train, 'w') as file:
    for data in train_list:
        json.dump(data, file)
        file.write('\n') 

train = "train.jsonl"

# Open the file in write mode and write each dictionary as a JSON object on a separate line
with open(train, 'w', encoding='utf-8') as file:
    for data in train_list:
        json.dump(data, file, ensure_ascii=False)
        file.write(' n')

# %%
df2 = pd.read_json('/home/zengwesley/NTUKaggle/team_dev.json')
eval_list=[]
for i in range(0, len(df2)):
    sample={"messages": [{"role": "system", "content": "You are a NLP researcher. This is am argument mining task. You should determine the given two sentence is support each other with output 1, attack each other with output 2, none with output 0. Only output one numeric number. "} , 
                         {"role": "user", "content": 'Argument1:'+str(df2.loc[i][1])+"\n\n"+'Argument2:'+str(df2.loc[i][2])}, 
                         {"role": "assistant", "content": "0"}]}
 
    eval_list.append(sample)

# %%
eval = "eval.jsonl"

# Open the file in write mode and write each dictionary as a JSON object on a separate line
#Use a txt file to read the api key
with open(eval, 'w') as file:
    for data in eval_list:
        json.dump(data, file)
        file.write('\n')  

# %%
from openai.types import FileContent, FileDeleted, FileObject

# %%
from openai import OpenAI
client = OpenAI(api_key = my_key)

# %%
train_file=client.files.create(
  file=open("train.jsonl", "rb"),
  purpose='fine-tune'
)
train_file # find the file id (FileObject(id='file-mX6ovaFhXt9wOhjLaRnwOyXs',)

# %%
eval_file=client.files.create(
  file=open("eval.jsonl", "rb"),
  purpose='fine-tune'
)
eval_file# find the file id (FileObject(id=another id,)

# %%
client.fine_tuning.jobs.create(
  training_file='file-mX6ovaFhXt9wOhjLaRnwOyXs', #train file id
  validation_file="file-7FoFNd2ItlqxlZX581ekecwz", #eval file id
  model="gpt-3.5-turbo-0125"
)


