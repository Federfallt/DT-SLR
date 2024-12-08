from openai import OpenAI

api_key="xxxx"

model_name = 'gpt-4o'  
temperature = 0.7

client = OpenAI(api_key=api_key)

with open('prompt.txt', 'r', encoding='utf-8') as pt:
    body = pt.readline()

with open('words.txt', 'r', encoding='utf-8') as input_file:
    words = input_file.readlines()
    for word in words:
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

        with open('description.txt', 'a', encoding='utf-8') as output_file:
            output_file.write(sens)
            output_file.write('\n')
