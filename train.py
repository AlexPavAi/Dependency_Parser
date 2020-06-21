import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from Models import BaseNet, AdvancedNet, nll_loss, paper_loss
from torch import optim
from data_loader import DpDataset
from torch.utils.data import DataLoader
from inference import compute_uas


os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(loss_type, attn_type, dropout_a, attn_dropout, EPOCHS, counter_fig, learning_rate):
    train_loss_array = []
    train_UAS_array = []
    test_UAS_array = []
    print_iter = 100
    test_epoch = 1
    print(f'device is: {device}')

    train_dataset = DpDataset('data', 'train', word_embeddings_name="glove.6B.100d")
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = DpDataset('data', 'test', vocab_dataset=train_dataset)
    test_loader = DataLoader(test_dataset, shuffle=False)
    model: AdvancedNet = AdvancedNet(word_emb_dim=100, tag_emb_dim=100, lstm_hidden_dim=125,
                                     attn_type=attn_type, attn_hidden_dim=100, attn_dropout=attn_dropout,
                                     word_vocab_size=len(train_dataset.word_idx_mappings),
                                     tag_vocab_size=len(train_dataset.pos_idx_mappings),
                                     appearance_count=train_dataset.word_idx_to_appearance, dropout_a=dropout_a,
                                     unk_word_ind=train_dataset.unk_word_idx,
                                     pre_trained_word_embedding=train_dataset.word_embeddings)

    use_cuda = torch.cuda.is_available()

    # if use_cuda:
    #     model.cuda()
    #     print("running of GPU")

    # else:
    #     print("running of CPU")

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
        if (epoch + 1) % test_epoch == 0:
            model.eval()

            num_correct_train = 0
            num_total_train = 0
            for i, input_data in enumerate(train_loader):
                words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
                true_heads = true_heads.squeeze(0)
                num_total_train += true_heads.shape[0]
                scores = model(words_idx_tensor, pos_idx_tensor)
                _, curr_num_correct = compute_uas(scores, true_heads)
                num_correct_train += curr_num_correct
            print("UAS on train:", num_correct_train / num_total_train)
            train_UAS_array.append(num_correct_train / num_total_train)

            num_correct_test = 0
            num_total_test = 0
            for i, input_data in enumerate(test_loader):
                words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
                true_heads = true_heads.squeeze(0)
                num_total_test += true_heads.shape[0]
                scores = model(words_idx_tensor, pos_idx_tensor)
                _, curr_num_correct = compute_uas(scores, true_heads)
                num_correct_test += curr_num_correct
            print("UAS on test:", num_correct_test/num_total_test)
            test_UAS_array.append(num_correct_test/num_total_test)
            model.train()
    test_uas = test_UAS_array[-1]
    test_uas_short = float("{:.3f}".format(test_uas))
    file_name = str(loss_type)+'_loss_'+str(attn_type)+'_attention_'+str(int(dropout_a * 100))+'_drop_a_'+str(int(attn_dropout * 100))+'_attn_dropout_'+str(EPOCHS)+'_EPOCHS_'
    plt.figure(counter_fig)
    plt.plot(train_loss_array)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('loss over epochs')
    plt.savefig(file_name + '_loss_over_epochs_')

    plt.figure(counter_fig + 1)
    plt.plot(test_UAS_array,label="test")
    plt.ylabel('UAS')
    plt.xlabel('epochs')
    plt.legend(file_name+'_UAS of test')
    plt.plot(train_UAS_array,label="train")
    plt.legend('UAS of train')
    plt.title('UAS over epochs. test UAS is: '+ str(test_uas_short))
    plt.savefig(file_name+'_UAS over epochs')

    return test_uas_short

attn_types =['additive', 'multiplicative']
# attn_type='additive'
loss_types = ['nll','paper']
# loss_types = ['nll']
# loss_type='paper'
dropouts_a = [1, 2, 5]
dropouts_attention = [0,0.25,0.5]
# attn_dropout = 0
# dropout_a = 0
EPOCHS = 25
learning_rate = 0.005
counter_fig = 1
max_UAS = 0
best_values = [0,0,0,0]
for loss_type in loss_types:
    for attn_type in attn_types:
        for dropout_a in dropouts_a:
            for attn_dropout in dropouts_attention:
                test_aus = test_uas_short = train(loss_type,attn_type,dropout_a,attn_dropout, EPOCHS, counter_fig, learning_rate)
                if test_aus > max_UAS:
                    max_UAS = test_aus
                    best_values = [loss_type, attn_type, dropout_a, attn_dropout]
                counter_fig += 2

print(f'best test UAS: {max_UAS}')
print("best values:")
for value in best_values:
    print(f'{value}')