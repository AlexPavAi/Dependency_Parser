import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Models import BaseNet, AdvancedNet, nll_loss, paper_loss
from torch import optim
from data_loader import DpDataset
from torch.utils.data import DataLoader
from inference import compute_uas
from inference import infer_heads
from misc import save_obj, load_obj

num_sentences = 5000
data_folder = 'data'
data_file_train = 'train'
data_file_comp = 'comp'
train_dataset = DpDataset(data_folder,data_file_train, word_embeddings_name="glove.6B.100d")
train_loader = DataLoader(train_dataset, shuffle=False)

comp_dataset = DpDataset(data_folder,data_file_comp, word_embeddings_name="glove.6B.100d")
comp_loader = DataLoader(comp_dataset, shuffle=False)

dropout_a = 2
attn_dropout = 0.5
# attn_dropout = 0
# dropout_a = 0
EPOCHS = 2
learning_rate = 0.005
attn_type='additive'
# loss_types = ['nll','paper']
loss_type= 'nll'

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = AdvancedNet(word_emb_dim=100, tag_emb_dim=100, lstm_hidden_dim=125,
                                     attn_type=attn_type, attn_hidden_dim=100, attn_dropout=attn_dropout,
                                     word_vocab_size=len(train_dataset.word_idx_mappings),
                                     tag_vocab_size=len(train_dataset.pos_idx_mappings),
                                     appearance_count=train_dataset.word_idx_to_appearance, dropout_a=dropout_a,
                                     unk_word_ind=train_dataset.unk_word_idx,
                                     pre_trained_word_embedding=train_dataset.word_embeddings)


if torch.cuda.device_count() > 1:
    print("Running on", torch.cuda.device_count(), "GPUs.")
    model = nn.DataParallel(model)
else:
    print("Running on single GPU.")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
PATH = "BEST_MODEL.PTH"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'], strict = False)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

inferred_head_all = np.zeros((num_sentences,1), dtype='object')
for i, input_data in enumerate(comp_loader):
    words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
    true_heads = true_heads.squeeze(0)

    scores = model(words_idx_tensor, pos_idx_tensor)
    infered_heads = infer_heads(scores)
    inferred_head_all[i,0] = infered_heads
    assert ((true_heads.shape[0]) == infered_heads.shape[0])

    # usa = compute_uas(scores, true_heads, squeeze=True)
save_obj(inferred_head_all,'inferred_heads_comp')
print("file saved")
print("")



print("")
