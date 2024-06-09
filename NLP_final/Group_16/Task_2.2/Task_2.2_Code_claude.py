#%%
import json
import yaml
import pandas as pd #update the latest openai version
import anthropic
import time

#%% [markdown]
# Load API key
with open("api_keys.yaml", "r") as file:
    data = yaml.safe_load(file)
for api_key in data['api_keys']:
    if 'claude3_key' in api_key:
        Claude_key = api_key['claude3_key']
client = anthropic.Anthropic( api_key = Claude_key)

# %% [markdown] 
# Prompt 1: Generate next 300 Response and save to raw.txt via Opus model
with open('dev.json') as f:
    data = json.load(f)
count = 0
with open('raw_result_claude_next300.txt', "w") as file:
    for item in data:
        count += 1
        if count > 300:
            sentence1 = item[0]
            sentence2 = item[1]
            prompt = "User input: Sentence 1:"+ sentence1+ "Sentence 2: "+sentence2
            
            message = client.messages.create(
                model="claude-3-opus-20240229", # 模型型號
                max_tokens=1000, # 選用，回傳token的最大長度，避免爆預算
                system="You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number(0,1,2).",
                messages=[
                    {"role": "user", "content":prompt}
                ]
            )

            print(message.content[0])
            message_to_save = str(message.content[0])+'\n'
            file.write(message_to_save)
            time.sleep(5)

# %% [markdown] 
# Prompt 1: Generate Response using Haiku Model
with open('dev.json') as f:
    data = json.load(f)
count = 0
with open('raw_result_claude_haiku.txt', "w") as file:
    for item in data:
        sentence1 = item[0]
        sentence2 = item[1]
        prompt = "User input: Sentence 1:"+ sentence1+ "Sentence 2: "+sentence2
        
        message = client.messages.create(
            model="claude-3-haiku-20240307", # 模型型號
            max_tokens=200, # 選用，回傳token的最大長度，避免爆預算
            system="You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number(0,1,2).",
            messages=[
                {"role": "user", "content":prompt}
            ]
        )

        print(message.content[0])
        message_to_save = str(message.content[0])+'\n'
        file.write(message_to_save)
        time.sleep(5)
        
# %% [markdown] 
# Prompt 2: Generate Response and save to raw.txt (Opus)
           
with open('dev.json') as f:
    data = json.load(f)
with open('raw_result2_claude_haiku.txt', "w") as file:
    for item in data:
        sentence1 = item[0]
        sentence2 = item[1]
        prompt = "User input: Sentence 1:"+ sentence1+ "Sentence 2: "+sentence2
        
        message = client.messages.create(
            model="claude-3-haiku-20240307", # 模型型號
            max_tokens=200, # 選用，回傳token的最大長度，避免爆預算
            system="You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number(0,1,2). Learn from the following examples, Sentence 1: Eating vegetables is good for health. Sentence 2: Vegetables contain important nutrients. Classification: 1 Sentence 1: I think there is a tendency in this industry to call everything new the next computer platform. Sentence 2: However, that said, I think AR can be huge. Classification: 2 Sentence 1: In terms of why we are withholding royalties, you cannot pay something when there is a dispute about the amount. Sentence 2: It is not 2013 it has nothing do with the display or the Touch ID or a gazillion other innovations that Apple has done. Classification: 0",
            messages=[
                {"role": "user", "content":prompt}
            ]
        )

        print(message.content[0])
        message_to_save = str(message.content[0])+'\n'
        file.write(message_to_save)
        time.sleep(5)


#%% [markdown]
# Process raw result and save to new text file
import regex as re
gpt_result = []
with open('raw_result2_claude_haiku.txt', "r") as file:
    content = file.readlines()

for line in content:
    match = re.search(r"text='.*?(\d+)'", line)
    if match:
        content_value = match.group(1)
        gpt_result.append(content_value)
    else:
        gpt_result.append('1')
        print("Content value not found.")

with open('converted_result2_claude_haiku.txt', "w") as file:
    my_string = '\n'.join(map(str, gpt_result))
    file.write(my_string)


# %% [markdown]
# Calculate F1 score for 1st prompt
from sklearn.metrics import f1_score
gt_result = []
with open('dev.json') as f:
    test_data = json.load(f)

for item in test_data:
    gt_result.append(item[2])
with open('converted_result2_claude_haiku.txt') as f:
    gpt_result = [int(line.strip()) for line in f]
    
gpt_result = [int(item) for item in gpt_result]
gt_result = [int(item) for item in gt_result]

micro_f1 = f1_score(gt_result, gpt_result, average='micro')
print(f'Micro F1 Score: {micro_f1}')

macro_f1 = f1_score(gt_result, gpt_result, average='macro')
print(f'Macro F1 Score: {macro_f1}')

weighted_f1 = f1_score(gt_result, gpt_result, average='weighted')
print(f'Weighted F1 Score: {weighted_f1}')



# %%
