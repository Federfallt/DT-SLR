# DT-SLR
This repo holds codes of the paper: Hierarchical Description Tree Search for Continuous Sign Language Recognition

sentence_cluster包含生成gloss的描述以及进行聚类得到树结点  
prototype_set.py选取结点的prototype，并生成文本特征  
update_matrix.py生成用于更新logits的矩阵  
tree_network.py是网络的主体，包括HDT搜索和更新过程  
