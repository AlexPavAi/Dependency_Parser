from torch import nn
from torch.functional import F
import torch

class net_prototype(nn.Module):
    def __init__(self, word_emb_dim, tag_emb_dim, lstm_hidden_dim, mlp_hidden_dim, word_vocab_size, tag_vocab_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)  # (B, len(sentence))
        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim)    # (B, len(sentence))
        self.lstm = nn.LSTM(input_size=word_emb_dim + tag_emb_dim, hidden_size=lstm_hidden_dim,
                            batch_first=True, bidirectional=True)  # (B, len(sentence), 2 * hidden)
        self.layer1_head = nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim)  # (B, mlp_hidden_dim ,len(sentence))
        self.layer1_modifier = nn.Linear(2 * lstm_hidden_dim, mlp_hidden_dim)  # (B, mlp_hidden_dim ,len(sentence))
        self.out_layer = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, word_idx, tag_idx):
        word_embeds = self.word_embedding(word_idx.to(self.device))
        tag_embeds = self.tag_embedding(tag_idx.to(self.device))
        x = word_embeds + tag_embeds
        lstm_out, _ = self.lstm(x)
        vh = self.layer1_head(lstm_out)
        vm = self.layer1_modifier(lstm_out)
        vh = vh.repeat(1, vh.shape[1], 1).view(1, vh.shape[1], vh.shape[1], -1)
        vm = vm.repeat(1, vm.shape[1], 1).view(1, vm.shape[1], vm.shape[1], -1)
        vm = vm.transpose(1, 2)
        out = vh + vm
        out = self.out_layer(out)
        out = out[:, :, 1:]
        return out

