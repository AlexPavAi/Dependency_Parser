import torch
from BaseNet import BaseNet, nll_loss
from torch import optim
from data_loader import PosDataset
from torch.utils.data import DataLoader

def train():
    EPOCHS = 15

    train_dataset = PosDataset('data', 'train')
    train_loader = DataLoader(train_dataset, shuffle=True)
    model = BaseNet(word_emb_dim=100, tag_emb_dim=100, lstm_hidden_dim=125, mlp_hidden_dim=100,
                    word_vocab_size=len(train_dataset.word_idx_mappings),
                    tag_vocab_size=len(train_dataset.pos_idx_mappings))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)

    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    epochs = EPOCHS
    for epoch in range(epochs):
        printable_loss = 0  # To keep track of the loss value
        i = 0
        for batch_idx, input_data in enumerate(train_loader):
            i += 1
            words_idx_tensor, pos_idx_tensor, sentence_length = input_data

            tag_scores = model(words_idx_tensor)
            tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)
            # print("tag_scores shape -", tag_scores.shape)
            # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
            loss = nll_loss(tag_scores, pos_idx_tensor.to(device))
            loss = loss / acumulate_grad_steps
            loss.backward()

            if i % acumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
            printable_loss += loss.item()
            _, indices = torch.max(tag_scores, 1)
            print(printable_loss)


train()
