from torch import nn
import torch

from inference import infer_heads


class WordDropout(nn.Module):
    def __init__(self, appearance_count, a=0.25, unk_ind=0):
        super().__init__()
        self.appearance_count = appearance_count
        self.a = float(a)
        self.unk_ind = unk_ind

    def forward(self, word_idx, train):
        if train and self.appearance_count is not None:
            out = word_idx.clone()
            p = self.a / (self.a + self.appearance_count[word_idx.squeeze(0)])
            drop_idx = torch.rand(word_idx.shape[1], requires_grad=False) < p
            out[0][drop_idx] = self.unk_ind
            return out
        return word_idx


class BaseNet(nn.Module):
    def __init__(self, word_vocab_size, tag_vocab_size, appearance_count=None, word_emb_dim=100, tag_emb_dim=100,
                 lstm_hidden_dim=125, mlp_hidden_dim=100, dropout_a=0.25, unk_word_ind=0, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.word_dropout = WordDropout(appearance_count, dropout_a, unk_word_ind)
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)  # (B, len(sentence))
        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim)    # (B, len(sentence))
        self.lstm = nn.LSTM(input_size=word_emb_dim + tag_emb_dim, hidden_size=lstm_hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)  # (B, len(sentence), 2 * hidden)
        self.layer1_head = nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim)  # (B, len(sentence), mlp_hidden_dim)
        self.layer1_modifier = nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim)  # (B, len(sentence), mlp_hidden_dim)
        self.out_layer = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, word_idx, tag_idx):
        self.word_dropout(word_idx, self.training)
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


class AdvancedNet(nn.Module):
    def __init__(self, word_vocab_size, tag_vocab_size, word_emb_dim=100, tag_emb_dim=100,
                 lstm_hidden_dim=125, lstm_dropout=0, mlp_hidden_dim=100, mlp_dropout=0, appearance_count=None,
                 dropout_a=0.25, unk_word_ind=0, pre_trained_word_embedding=None, freeze_word_embedding=True,
                 device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.word_dropout = WordDropout(appearance_count, dropout_a, unk_word_ind)
        if pre_trained_word_embedding is None:
            self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pre_trained_word_embedding, freeze=freeze_word_embedding)
        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim)    # (B, len(sentence))
        self.lstm = nn.LSTM(input_size=word_emb_dim + tag_emb_dim, hidden_size=lstm_hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.layer1_head = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim),
            nn.Dropout(p=mlp_dropout)
        )
        self.layer1_modifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim),
            nn.Dropout(p=mlp_dropout)
        )
        self.out_layer = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, word_idx, tag_idx):
        self.word_dropout(word_idx, self.training)
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


def paper_loss(out, true_heads):
    sentence_len = true_heads.shape[0]
    modifiers = torch.arange(sentence_len)
    true_score = torch.sum(out[:, true_heads, modifiers])
    scores = out + 1
    scores[:, true_heads, modifiers] -= 1
    inferred_heads = infer_heads(scores)
    inferred_score = torch.sum(out[:, inferred_heads, modifiers])
    loss = torch.max(torch.tensor(0), true_score - inferred_score + 1)
    return loss
