import torch
from BaseNet import BaseNet, nll_loss
from torch import optim
from data_loader import PosDataset
from torch.utils.data import DataLoader
from inference import compute_uas

def train():
    EPOCHS = 15
    print_iter = 10
    test_epoch = 3

    train_dataset = PosDataset('data', 'train')
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = PosDataset('data', 'test')
    test_loader = DataLoader(test_dataset, shuffle=False)
    model: BaseNet = BaseNet(word_emb_dim=100, tag_emb_dim=100, lstm_hidden_dim=125, mlp_hidden_dim=100,
                             word_vocab_size=len(train_dataset.word_idx_mappings),
                             tag_vocab_size=len(train_dataset.pos_idx_mappings))

    use_cuda = torch.cuda.is_available()

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
        for i, input_data in enumerate(train_loader):
            words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
            true_heads = true_heads.squeeze(0)

            scores = model(words_idx_tensor, pos_idx_tensor)
            loss = nll_loss(scores, true_heads.to(model.device))
            loss = loss / acumulate_grad_steps
            loss.backward()

            if (i+1) % acumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
                printable_loss += loss.item()

            if (i+1) % (acumulate_grad_steps * print_iter) == 0:
                print("epoch", epoch+1, "iter", (i+1)//acumulate_grad_steps, "loss:", printable_loss)
                printable_loss = 0
        if (epoch + 1) % test_epoch == 0:
            model.eval()
            num_correct = 0
            num_total = 0
            for i, input_data in enumerate(test_loader):
                words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
                true_heads = true_heads.squeeze(0)
                num_total += true_heads.shape[0]
                scores = model(words_idx_tensor, pos_idx_tensor)
                _, curr_num_correct = compute_uas(scores, true_heads)
                num_correct += curr_num_correct
            print("UAS on test:", num_correct/num_total)
            model.train()


train()
