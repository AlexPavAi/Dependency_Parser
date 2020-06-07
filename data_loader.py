import os
import torch
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset
from collections import Counter
from collections import defaultdict

UNKNOWN_TOKEN = "<unk>"
ROOT_TOKEN = "<ROOT>"  # Optional: this is used to pad a batch of sentences in different lengths.
SPECIAL_TOKENS = [UNKNOWN_TOKEN, ROOT_TOKEN]


def get_vocabs(file_path):
    """
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param file_path: full path of the corpuses
            Return:
              - word2idx
              - tag2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    with open(file_path) as f:
        for line in f:
            if line.strip():
                splited_words = line.split()
                # print(line)
                word = splited_words[1]
                pos_tag = splited_words[3]
                word_dict[word] += 1
                pos_dict[pos_tag] += 1

    index_dict_word = Vocab(Counter(word_dict), specials=SPECIAL_TOKENS)
    index_dict_pos = Vocab(Counter(pos_dict), specials=[ROOT_TOKEN])
    return word_dict, pos_dict, index_dict_word.stoi, index_dict_pos.stoi


class PosDataReader:
    def __init__(self, file):
        self.file = file
        # self.word_dict = word_dict
        # self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            cur_sentence = []
            for line in f:
                if line.strip():
                    splited_words = line.split()
                    # print(line)
                    word = splited_words[1]
                    pos_tag = splited_words[3]
                    head_index = splited_words[6]
                    cur_sentence.append((word, pos_tag, head_index))
                else:
                    self.sentences.append(cur_sentence)
                    cur_sentence = []

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class PosDataset(Dataset):
    def __init__(self, dir_path: str, subset: str,
                 padding=False, word_embeddings=None):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        # self.file = dir_path + subset + ".labeled"
        self.file = os.path.join(dir_path, subset) + ".labeled"
        self.datareader = PosDataReader(self.file)
        # self.vocab_size = len(self.datareader.word_dict)
        _, _, self.word_idx_mappings, self.pos_idx_mappings = get_vocabs(self.file)
        self.sentences_dataset = self.convert_sentences_to_dataset()

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, head, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, head, sentence_len

    def convert_sentences_to_dataset(self):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_head_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = [self.word_idx_mappings[ROOT_TOKEN]]
            pos_idx_list = [self.pos_idx_mappings[ROOT_TOKEN]]
            head_idx_list = []
            for word, pos, head in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
                head_idx_list.append(head)
            sentence_len = len(words_idx_list)

            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_head_list,
                                                                     sentence_len_list))}


def main():
    data_dir = "data"
    # get_vocabs(list_of_pathes)
    dir_path = "data"
    str = "train"
    train = PosDataset(data_dir, str)
    print("")


if __name__ == '__main__':
    main()
