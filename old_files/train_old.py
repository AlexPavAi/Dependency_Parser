import torch
import os
import matplotlib.pyplot as plt
from code_directory.Models import AdvancedNet, regularized_paper_loss
from torch import optim
from code_directory.data_loader import DpDataset
from torch.utils.data import DataLoader
from code_directory.inference import compute_uas


os.environ["CUDA_VISIBLE_DEVICES"]='2,3,6'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


def train():
    torch.manual_seed(0)
    train_loss_array = []
    train_UAS_array = []
    test_UAS_array = []
    EPOCHS = 15
    print_iter = 20
    test_epoch = 3

    train_dataset = DpDataset('../code_directory/data', 'train', word_embeddings_name="glove.6B.100d")
    # train_dataset = DpDataset('data', 'train', word_embeddings_name=None)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = DpDataset('../code_directory/data', 'test', vocab_dataset=train_dataset)
    test_loader = DataLoader(test_dataset, shuffle=False)
    model: AdvancedNet = AdvancedNet(word_emb_dim=100, tag_emb_dim=100, lstm_hidden_dim=125, lstm_out_dropout=0.,
                                     lstm_dropout=0.1,
                                     attn_type='multiplicative', attn_hidden_dim=100, attn_dropout=0.25,
                                     word_vocab_size=len(train_dataset.word_idx_mappings),
                                     tag_vocab_size=len(train_dataset.pos_idx_mappings),
                                     appearance_count=train_dataset.word_idx_to_appearance, dropout_a=5,
                                     unk_word_ind=train_dataset.unk_word_idx,
                                     pre_trained_word_embedding=train_dataset.word_embeddings)

    # model: TransformerModel = TransformerModel(word_vocab_size=len(train_dataset.word_idx_mappings),
    #                                            tag_vocab_size=len(train_dataset.pos_idx_mappings),
    #                                            appearance_count=train_dataset.word_idx_to_appearance,
    #                                            unk_word_ind=train_dataset.unk_word_idx,
    #                                            pre_trained_word_embedding=train_dataset.word_embeddings,
    #                                            word_emb_dim=100, tag_emb_dim=100,
    #                                            nhead=8, transformer_hidden=256, transformer_layers=6,
    #                                            transformer_dropout=0.1,
    #                                            attn_type='multiplicative', attn_hidden_dim=100, attn_dropout=0.1,
    #                                            dropout_a=5, freeze_word_embedding=True)

    use_cuda = torch.cuda.is_available()

    # if use_cuda:
    #     model.cuda()
    #     print("running of GPU")

    # else:
    #     print("running of CPU")
    model.to(device)

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)

    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [9], gamma=0.2)
    # optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.)
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
            # loss = nll_loss(scores.to("cpu"), true_heads)
            # loss = paper_loss(scores.to("cpu"), true_heads)
            # loss = nll_loss(scores.to("cpu"), true_heads) + paper_loss(scores.to("cpu"), true_heads)
            loss = regularized_paper_loss(scores.to("cpu"), true_heads, alpha=0.5)
            # if epoch + 1 < 4:
            #     loss = paper_loss(scores.to("cpu"), true_heads)
            # else:
            #     loss = variational_paper_loss(scores.to("cpu"), true_heads, std=1.)

            loss = loss / acumulate_grad_steps
            loss.backward()

            if (i+1) % acumulate_grad_steps == 0:
                optimizer.step()
                scheduler.step()
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

    plt.figure(1)
    plt.plot(train_loss_array)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('loss over epochs')
    plt.savefig('loss over epochs')

    plt.figure(2)
    plt.plot(test_UAS_array,label="test")
    plt.ylabel('UAS')
    plt.xlabel('epochs')
    plt.legend('UAS of test')
    plt.plot(train_UAS_array,label="train")
    plt.legend('UAS of train')
    plt.title('UAS over epochs')
    plt.savefig('UAS over epochs')


train()