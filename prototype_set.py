import pandas as pd
import torch
from transformers import CLIPTextModel, CLIPTokenizer

df = pd.read_csv('./sentence_cluster/description.csv')

clip_path = "./pretrained/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True, torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True, torch_dtype=torch.float16).to('cuda')


l1 = df['l1_label'].max() + 1

text_inputs = [df[df['l1_label'] == k].iloc[0]['sentence'] for k in range(l1)]
token = tokenizer(text_inputs, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
l1_prototype = text_encoder(token.input_ids.to("cuda"))[0].half() # l1 x 77 x 768
l1_prototype = l1_prototype.detach()
torch.save(l1_prototype, "./HDT_prototype/l1.pt")

l2_max = df['l2_label'].max() + 1
padded_tf_list = []
for i in range(l1):
    subset_df = df[df['l1_label'] == i]

    l2 = subset_df['l2_label'].max() + 1

    text_inputs = [subset_df[subset_df['l2_label'] == k].iloc[0]['sentence'] for k in range(l2)]
    token = tokenizer(text_inputs, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_feature = text_encoder(token.input_ids.to("cuda"))[0].half() # l2 x 77 x 768

    padded_tf = torch.nn.functional.pad(text_feature, (0, 0, 0, 0, 0, l2_max - l2)) # l2_max x 77 x 768
    padded_tf_list.append(padded_tf)

l2_prototype = torch.cat(padded_tf_list, dim=0).view(l1, l2_max, 77, 768) # l1 x l2_max x 77 x 768
l2_prototype = l2_prototype.detach()
torch.save(l2_prototype, "./HDT_prototype/l2.pt")
