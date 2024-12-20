import pandas as pd
import torch

target = 'phoenix2014' # phoenix2014T, phoenix2014, CSLDaily

df = pd.read_csv('./sentence_cluster/description_{}.csv'.format(target))

num_classes = 1296 # phoenix2014T: 1116, phoenix2014: 1296, CSLDaily: 2001

l1 = df['l1_label'].max() + 1

ls_list = []
for i in range(l1):
    subset_df = df[df['l1_label'] == i]
    ls_row = torch.zeros((1, num_classes))
    for idx in subset_df['word_index']:
        ls_row[0, idx+1] = 1
    ls_list.append(ls_row)

ls_matrix = torch.cat(ls_list, dim=0).view(num_classes, -1) # n x l1
sums = ls_matrix.sum(dim=1, keepdim=True)
sums[sums == 0] = 1
ls_matrix = ls_matrix / sums
torch.save(ls_matrix, "./HDT_prototype/ls_matrix_{}.pt".format(target))
