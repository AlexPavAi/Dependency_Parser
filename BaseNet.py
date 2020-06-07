from torch import nn
import torch


class WordDropout(nn.Module):
    def __init__(self, device, p=0.1, unk_ind=0):
        super().__init__()
        self.p = p
        self.unk_ind = unk_ind
        self.device = device

    def forward(self, word_idx):
        drop_idx = torch.rand(word_idx.shape, device=self.device) < self.p
        word_idx[drop_idx] = self.unk_ind


class BaseNet(nn.Module):
    def __init__(self, word_emb_dim, tag_emb_dim, lstm_hidden_dim, mlp_hidden_dim, word_vocab_size, tag_vocab_size,
                 word_dropout=0.1, unk_ind=0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_dropout = WordDropout(p=word_dropout, unk_ind=unk_ind, device=self.device)
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)  # (B, len(sentence))
        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim)    # (B, len(sentence))
        self.lstm = nn.LSTM(input_size=word_emb_dim + tag_emb_dim, hidden_size=lstm_hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)  # (B, len(sentence), 2 * hidden)
        self.layer1_head = nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim)  # (B, len(sentence), mlp_hidden_dim)
        self.layer1_modifier = nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim)  # (B, len(sentence), mlp_hidden_dim)
        self.out_layer = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, word_idx, tag_idx):
        word_embeds = self.word_embedding(word_idx.to(self.device))
        tag_embeds = self.tag_embedding(tag_idx.to(self.device))
        x = torch.cat((word_embeds, tag_embeds), dim=2)
        lstm_out, _ = self.lstm(x)
        vh = self.layer1_head(lstm_out)
        vm = self.layer1_modifier(lstm_out)
        vh = vh.repeat(1, vh.shape[1], 1).view(vh.shape[0], vh.shape[1], vh.shape[1], -1)
        vh = vh.transpose(1, 2)
        vm = vm.repeat(1, vm.shape[1], 1).view(vm.shape[0], vm.shape[1], vm.shape[1], -1)
        out = vh + vm
        out = torch.tanh(out)
        out = self.out_layer(out).squeeze(3)
        out = out[:, :, 1:]
        return out


def nll_loss(out, true_heads):
    sentence_len = true_heads.shape[0]
    true_scores = out[:, true_heads, torch.arange(sentence_len)]
    sum_exp = torch.sum(torch.exp(out), dim=1)
    log_sum_exp = torch.log(sum_exp)
    return torch.mean(- true_scores + log_sum_exp)
