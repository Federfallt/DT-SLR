# from openai import OpenAI
import requests
import json

target = 'CSLDaily' # phoenix2014T, phoenix2014, CSLDaily

'''
api_key="xxxx"

model_name = 'gpt-4o'  
temperature = 0.7

client = OpenAI(api_key=api_key)

with open('prompt_{}.txt'.format(target), 'r', encoding='utf-8') as pt:
    body = pt.readline()

with open('words_{}.txt'.format(target), 'r', encoding='utf-8') as input_file:
    words = input_file.readlines()
    for idx, word in enumerate(words):
        prompt = body + word

        completion = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    
        sens = completion.choices[0].message.content

        with open('description_{}.txt'.format(target), 'a', encoding='utf-8') as output_file:
            output_file.write(sens)
            output_file.write('\n')
            output_file.write('\n')

        print("generating gloss" + str(idx+1))
'''

url = "https://oa.api2d.net/v1/chat/completions"

headers = {
   'Authorization': 'Bearer xxx',
   'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
   'Content-Type': 'application/json'
}

with open('prompt_{}.txt'.format(target), 'r', encoding='utf-8') as pt:
    body = pt.readline()

with open('words_{}.txt'.format(target), 'r', encoding='utf-8') as input_file:
    words = input_file.readlines()
    for idx, word in enumerate(words):
        prompt = body + word

        payload = json.dumps({
        "model": "gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": prompt
        }
        ],
        "safe_mode": False,
        })
    
        response = requests.request("POST", url, headers=headers, data=payload)

        sens = response.json()["choices"][0]["message"]["content"]

        with open('description_{}.txt'.format(target), 'a', encoding='utf-8') as output_file:
            output_file.write(sens)
            output_file.write('\n')
            output_file.write('\n')

        print("generating gloss" + str(idx+1))
