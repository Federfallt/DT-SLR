import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        #self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

        self.vf = nn.Linear(hidden_size, 768)
        self.tf = nn.Linear(77*768, 768)
        self.l1 = torch.load("./HDT_prototype/l1.pt").cuda()
        self.l2 = torch.load("./HDT_prototype/l2.pt").cuda()
        self.up = torch.load("./HDT_prototype/up_matrix.pt").cuda()

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def Hierarchical_updating(self, l1_index, l2_index_list):
        up_list = []
        for i, idx in enumerate(l1_index):
            up_row = self.up[idx, l2_index_list[i], :].view(1, -1) # 1 x n
            up_list.append(up_row)
        up_matrix = torch.cat(up_list, dim=0) # T x n
        return up_matrix

    def HDT_search(self, visual_feature):
        v_f = self.vf(visual_feature) # T x C'
        t_f = self.tf(self.l1) # l1 x C'
        normalized_visual = F.normalize(v_f, dim=1)
        normalized_textual = F.normalize(t_f, dim=1)

        l1_similarity = torch.matmul(normalized_visual, normalized_textual.T) # T x l1
        _, l1_index = torch.max(l1_similarity, dim=1) # T

        l2_index_list = []
        for i, idx in enumerate(l1_index):
            t_f = self.tf(self.l2[idx.item()]) # l2_max x C'
            normalized_textual = F.normalize(t_f, dim=1)

            l2_similarity = torch.matmul(normalized_visual[i, :].unsqueeze(0), normalized_textual.T) # 1 x l2_max
            _, l2_index = torch.max(l2_similarity, dim=1) # 1

            l2_index_list.append(l2_index.item())

        up_matrix = self.Hierarchical_updating(l1_index, l2_index_list)

        return up_matrix

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * temp, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])

        up_matrix = self.HDT_search(tm_outputs['predictions'])
        up_outputs = torch.mul(outputs, up_matrix)

        pred = None if self.training \
            else self.decoder.decode(up_outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            #"visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": up_outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "loss_LiftPool_u": conv1d_outputs['loss_LiftPool_u'],
            "loss_LiftPool_p": conv1d_outputs['loss_LiftPool_p'],
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        total_loss = {}
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                total_loss['ConvCTC'] = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += total_loss['ConvCTC']
            elif k == 'SeqCTC':
                total_loss['SeqCTC'] = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
                loss += total_loss['SeqCTC']
            elif k == 'Dist':
                total_loss['Dist'] = weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
                loss += total_loss['Dist']
            elif k == 'Cu':
                total_loss['Cu'] = weight * ret_dict["loss_LiftPool_u"]
                loss += total_loss['Cu']
            elif k == 'Cp':
                total_loss['Cp'] = weight * ret_dict["loss_LiftPool_p"]   
                loss += total_loss['Cp'] 
        return loss, total_loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
