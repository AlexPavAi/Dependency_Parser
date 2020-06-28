import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from code_directory.Models import BaseNet, AdvancedNet, nll_loss, regularized_paper_loss
from torch import optim
from code_directory.data_loader import DpDataset
from torch.utils.data import DataLoader

from code_directory.eval import eval_model


def train(epochs, model_type='advanced', test_epoch=1, save_model=True, model_path='model.pkl',
          save_plots=False, plot_dir='./', checkpoint_at_test=False, checkpoint_path=None, time_run=False):
    """
    :param epochs: number of epochs
    :param model_type: type of the model 'advanced' or 'base'
    :param test_epoch: test every test_epoch epochs
    :param save_model: save the model or not
    :param model_path: path to save the model
    :param save_plots: save the plot
    :param plot_dir: directory to save the plots in
    :param checkpoint_at_test: if True saves checkpoint of the model every test
    :param checkpoint_path: path to save the checkpoint to (appended the epoch number) if checkpoint_at_test==True
    :param time_run: if True rimes the run
    :return: the trained model
    """
    if time_run:
        t0 = time.time()
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_uas_array = []
    train_loss_array = []
    test_uas_array = []
    test_loss_array = []
    if model_type == 'advanced':
        train_dataset = DpDataset('data', 'train', word_embeddings_name="glove.6B.100d")
        model: AdvancedNet = AdvancedNet(word_emb_dim=100, tag_emb_dim=100, lstm_hidden_dim=125,
                                         attn_type='multiplicative', attn_hidden_dim=100, attn_dropout=0.25,
                                         lstm_dropout=0.1,
                                         word_vocab_size=len(train_dataset.word_idx_mappings),
                                         tag_vocab_size=len(train_dataset.pos_idx_mappings),
                                         appearance_count=train_dataset.word_idx_to_appearance, dropout_a=5,
                                         unk_word_ind=train_dataset.unk_word_idx,
                                         pre_trained_word_embedding=train_dataset.word_embeddings)
        # dropout_a is the alpha for word dropout
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [6], gamma=0.2)
        loss_func = lambda out, th: regularized_paper_loss(out, th, alpha=0.5)
    if model_type == 'base':
        train_dataset = DpDataset('data', 'train', word_embeddings_name=None)
        model: BaseNet = BaseNet(word_emb_dim=100, tag_emb_dim=25, lstm_hidden_dim=125,
                                 word_vocab_size=len(train_dataset.word_idx_mappings),
                                 tag_vocab_size=len(train_dataset.pos_idx_mappings),
                                 appearance_count=train_dataset.word_idx_to_appearance, dropout_a=0.25,
                                 unk_word_ind=train_dataset.unk_word_idx)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = None
        loss_func = nll_loss
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_dataset = DpDataset('data', 'test', vocab_dataset=train_dataset)
    test_loader = DataLoader(test_dataset, shuffle=False)
    model.to(device)
    acumulate_grad_steps = 50
    print("Training Started")
    for epoch in range(epochs):
        printable_loss = 0
        for i, input_data in enumerate(train_loader):
            words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
            true_heads = true_heads.squeeze(0)
            scores = model(words_idx_tensor, pos_idx_tensor)
            loss = loss_func(scores.to("cpu"), true_heads)
            loss = loss / acumulate_grad_steps
            loss.backward()

            if (i+1) % acumulate_grad_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                printable_loss += loss.item()

        if (epoch + 1) % test_epoch == 0:
            train_uas, train_loss = eval_model(model, train_loader, loss_func, uas_list=train_uas_array,
                                               loss_list=train_loss_array)
            test_uas, test_loss = eval_model(model, test_loader, loss_func, uas_list=test_uas_array,
                                             loss_list=test_loss_array)
            print("Epoch {} Completed,\tTrain Loss: {}, \tTest Loss: {},\tTrain UAS: {}\t Test UAS: {}".format(
                epoch + 1, train_loss, test_loss, train_uas, test_uas
            ))
            model.train()
            if checkpoint_at_test:
                torch.save({'state_dict': model.state_dict(), 'args': model.args,
                            'test_uas': test_uas, 'train_uas': train_uas,
                            'indexing_dictionaries': (train_dataset.word_idx_mappings, test_dataset.pos_idx_mappings,
                                                      train_dataset.word_idx_to_appearance, None)
                            }, checkpoint_path+'_'+str(epoch+1))
        else:
            print("Epoch {} Completed,\tTrain Loss: {}".format(
                epoch + 1, printable_loss * acumulate_grad_steps / len(train_loader)
            ))
    if save_model:
        torch.save({'state_dict': model.state_dict(), 'args': model.args,
                    'test_uas_arr': test_uas_array, 'train_uas_arr': train_uas_array,
                    'test_loss_arr': test_loss_array, 'train_loss_arr': train_loss_array,
                    'indexing_dictionaries': (train_dataset.word_idx_mappings, test_dataset.pos_idx_mappings,
                                              train_dataset.word_idx_to_appearance, None)
                    }, model_path)
    if save_plots:
        epochs_arr = np.arange(1, epochs + 1, test_epoch)

        plt.figure()
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('loss over epochs')
        plt.plot(epochs_arr, train_loss_array, label="train")
        plt.plot(epochs_arr, test_loss_array, label="test")
        plt.savefig(plot_dir + 'loss_over_epochs.png')

        plt.figure()
        plt.ylabel('UAS')
        plt.xlabel('epochs')
        plt.plot(epochs_arr, train_uas_array, label="train")
        plt.plot(epochs_arr, test_uas_array, label="test")
        plt.legend()
        plt.savefig(plot_dir + 'UAS_over_epochs')
    if time_run:
        print('training took:', time.time()-t0)


if __name__ == '__main__':
    train(4, model_type='base', save_model=True, model_path="basic_model.pkl", time_run=True)
    train(18, model_type='advanced', save_model=True, model_path="advanced_model.pkl", time_run=True)


