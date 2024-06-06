#%%
import json
import yaml
import pandas as pd #update the latest openai version
import google.generativeai as genai
import time

# %% [markdown] 
# Read the training file
train_data = pd.read_json('test_claude.json')
train_data.head()

#%% [markdown]
# Load Gemini API key and Gemini Model
with open("api_keys.yaml", "r") as file:
    data = yaml.safe_load(file)
for api_key in data['api_keys']:
    if 'gemini' in api_key:
        GOOGLE_API_KEY = api_key['gemini']
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
try:
    model.generate_content(
      "test",
    )
    print("Set Gemini API KEY sucessfully!!")
except:
    print("There seems to be something wrong with your Gemini API. Please follow our demonstration in the slide to get a correct one.")

#%% [markdown]
# Genrate a response from Gemini and save response to .txt
with open('dev.json') as f:
    data = json.load(f)
with open('raw_result_gemini.txt', "w") as file:
    for item in data:
        response = model.generate_content("""
        You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, 
        else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. 
        Only output one numeric number(0,1,2).

        User input: Sentence 1: {item[0]} Sentence 2: {item[1]}
        """)
        print(response.text, item[0], item[1])
        message_to_save = response.text+'\n'
        file.write(message_to_save)
        time.sleep(5)
        

#%% [markdown]
# Use a fine-tuned GPT3 model for 1st prompt

from openai import OpenAI
client = OpenAI(api_key = openai_gpt_key)
message_to_save = []

with open('dev.json') as f:
    data = json.load(f)

with open('raw_result.txt', "w") as file:
    for item in data:
        completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:iis-academia-sinica::9RD2EKa7",
        messages=[
            {"role": "system", "content": "You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number(0,1,2)."},
            {"role": "user", "content":  'Sentence 1: '+item[0]+"\n\n"+'Sentence 2: '+item[1]}
        ]
        )
        tmp = completion.choices[0].message
        # print(tmp)
        message_to_save = str(tmp)+'\n'
        # print(message_to_save)
        file.write(message_to_save)



#%% [markdown]
# Process raw result and save to new text file
import regex as re
gpt_result = []
with open('raw_result.txt', "r") as file:
    content = file.readlines()

for line in content:
    match = re.search(r"content='(.*?)'", line)
    if match:
        content_value = match.group(1)
        # print(content_value)
        gpt_result.append(content_value)
    else:
        print("Content value not found.")

with open('converted_result.txt', "w") as file:
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

gpt_result = [int(item) for item in gpt_result]
gt_result = [int(item) for item in gt_result]
# print(gpt_result)
# print(gt_result)
micro_f1 = f1_score(gt_result, gpt_result, average='micro')
print(f'Micro F1 Score: {micro_f1}')

macro_f1 = f1_score(gt_result, gpt_result, average='macro')
print(f'Macro F1 Score: {macro_f1}')

weighted_f1 = f1_score(gt_result, gpt_result, average='weighted')
print(f'Weighted F1 Score: {weighted_f1}')



# %% [markdown]
# Use 2nd prompt for the same model
from openai import OpenAI
client = OpenAI(api_key = openai_gpt_key)
message_to_save = []

with open('dev.json') as f:
    data = json.load(f)
    
with open('raw_result2.txt', "w") as file:
    for item in data:
        completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:iis-academia-sinica::9RD2EKa7",
        messages=[
            {"role": "system", "content": 'You are a NLP researcher. This is an argument relation detection task. You should determine if there is a Supporting relation from sentence 1 to sentence 2 with output 1, else if it is an Attacking relation from sentence 1 to sentence 2 with output 2, or it does not detect any relationship between sentence 1 and sentence 2 with output 0. Only output one numeric number(0,1,2). Learn from the following examples, Sentence 1: Eating vegetables is good for health. Sentence 2: Vegetables contain important nutrients. Classification: 1 Sentence 1: I think there is a tendency in this industry to call everything new the next computer platform. Sentence 2: However, that said, I think AR can be huge. Classification: 2 Sentence 1: In terms of why we are withholding royalties, you cannot pay something when there is a dispute about the amount. Sentence 2: It is not 2013 it has nothing do with the display or the Touch ID or a gazillion other innovations that Apple has done. Classification: 0'}
            ,{"role": "user", "content":  'Sentence 1: '+item[0]+"\n\n"+'Sentence 2: '+item[1]}
        ]
        )
        tmp = completion.choices[0].message
        # print(tmp)
        message_to_save = str(tmp)+'\n'
        # print(message_to_save)
        file.write(message_to_save)


#%% [markdown]
# Process raw2 result and save to new text file
import regex as re
gpt_result2 = []
with open('raw_result2.txt', "r") as file:
    content = file.readlines()

for line in content:
    match = re.search(r"content='(.*?)'", line)
    if match:
        content_value = match.group(1)
        # print(content_value)
        gpt_result2.append(content_value)
    else:
        print("Content value not found.")

with open('converted_result2.txt', "w") as file:
    my_string = '\n'.join(map(str, gpt_result2))
    file.write(my_string)

# %% [markdown]
# Calculate F1 score for 2nd prompt
from sklearn.metrics import f1_score
gt_result = []
with open('dev.json') as f:
    test_data = json.load(f)

for item in test_data:
    gt_result.append(item[2])

gpt_result2 = [int(item) for item in gpt_result2]
gt_result = [int(item) for item in gt_result]
# print(gpt_result)
# print(gt_result)
micro_f1 = f1_score(gt_result, gpt_result2, average='micro')
print(f'Micro F1 Score: {micro_f1}')

macro_f1 = f1_score(gt_result, gpt_result2, average='macro')
print(f'Macro F1 Score: {macro_f1}')

weighted_f1 = f1_score(gt_result, gpt_result2, average='weighted')
print(f'Weighted F1 Score: {weighted_f1}')
# %%
