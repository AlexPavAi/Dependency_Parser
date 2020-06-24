import torch
import os
import torch.nn as nn
from Models import BaseNet, AdvancedNet, nll_loss, paper_loss
from torch import optim
from data_loader import DpDataset
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(loss_type, attn_type, dropout_a, attn_dropout, EPOCHS, learning_rate):
    PATH = "BEST_MODEL_RECENT.PTH"
    train_loss_array = []
    print_iter = 100
    print(f'device is: {device}')

    train_dataset = DpDataset('data', 'train', word_embeddings_name="glove.6B.100d")
    train_loader = DataLoader(train_dataset, shuffle=True)
    model: AdvancedNet = AdvancedNet(word_emb_dim=100, tag_emb_dim=100, lstm_hidden_dim=125,
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

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)

    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    epochs = EPOCHS
    for epoch in range(epochs):
        printable_loss = 0  # To keep track of the loss value
        for i, input_data in enumerate(train_loader):
            words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
            true_heads = true_heads.squeeze(0)

            scores = model(words_idx_tensor, pos_idx_tensor)
            # loss = nll_loss(scores, true_heads.to(model.device))
            if loss_type =='nll':
                loss = nll_loss(scores.to("cpu"), true_heads)
            else:
                loss = paper_loss(scores.to("cpu"), true_heads)

            loss = loss / acumulate_grad_steps
            loss.backward()

            if (i+1) % acumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
                printable_loss += loss.item()

            if (i+1) % (acumulate_grad_steps * print_iter) == 0:
                print("epoch", epoch+1, "iter", (i+1)//acumulate_grad_steps, "loss:", printable_loss)
                train_loss_array.append(printable_loss)
                printable_loss = 0
    # torch.save(model, PATH)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH)
    print("model has been saved")


# attn_types =['additive', 'multiplicative']
attn_type='multiplicative'
# loss_types = ['nll','paper']
# loss_type= 'nll'
loss_type='paper'
dropout_a = 5
attn_dropout = 0.25
# attn_dropout = 0
# dropout_a = 0
EPOCHS = 25
learning_rate = 0.005



train(loss_type,attn_type,dropout_a,attn_dropout, EPOCHS, learning_rate)
