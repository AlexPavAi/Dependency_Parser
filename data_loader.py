import os
import torch
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset
from collections import Counter
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader

UNKNOWN_TOKEN = "<unk>"
ROOT_TOKEN = "<ROOT>"  # Optional: this is used to pad a batch of sentences in different lengths.
SPECIAL_TOKENS = [UNKNOWN_TOKEN, ROOT_TOKEN]


def get_vocabs(file_path, from_other_dataset=None, word_embeddings_name=None):
    """
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param from_other_dataset: getting vocab from the dataset from_dataset
        :param file_path: full path of the corpuses
        :param word_embeddings_name: name pre trained word embedding wanted to use
            Return:
              - word2idx
              - tag2idx
              - word index to number of appearances
              - word vectors
    """
    if from_other_dataset is None:
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

        index_dict_word = Vocab(Counter(word_dict), specials=SPECIAL_TOKENS, vectors=word_embeddings_name)
        index_dict_pos = Vocab(Counter(pos_dict), specials=SPECIAL_TOKENS)
        word_idx_to_appearance = torch.zeros(len(index_dict_word.stoi), dtype=torch.float)
        for word in word_dict:
            word_idx_to_appearance[index_dict_word.stoi[word]] = word_dict.get(word, float('inf'))
        return index_dict_word.stoi, index_dict_pos.stoi, word_idx_to_appearance, index_dict_word.vectors
    else:
        return from_other_dataset.word_idx_mappings, from_other_dataset.pos_idx_mappings, \
               from_other_dataset.word_idx_to_appearance, None


class DpDataReader:
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
                    head_index = int(splited_words[6])
                    cur_sentence.append((word, pos_tag, head_index))
                else:
                    self.sentences.append(cur_sentence)
                    cur_sentence = []

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


class DpDataset(Dataset):
    def __init__(self, dir_path: str, subset: str, vocab_dataset=None,
                 word_embeddings_name=None):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        # self.file = dir_path + subset + ".labeled"
        self.file = os.path.join(dir_path, subset) + ".labeled"
        self.datareader = DpDataReader(self.file)
        # self.vocab_size = len(self.datareader.word_dict)
        self.word_idx_mappings, self.pos_idx_mappings, self.word_idx_to_appearance, self.word_embeddings = \
            get_vocabs(self.file, vocab_dataset, word_embeddings_name)

        self.unk_word_idx = self.word_idx_mappings[UNKNOWN_TOKEN]
        self.unk_pos_idx = self.pos_idx_mappings[UNKNOWN_TOKEN]
        self.sentences_dataset = self.convert_sentences_to_dataset()
        self.name = "here for debugging"

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
                words_idx_list.append(self.word_idx_mappings.get(word, self.unk_word_idx))
                pos_idx_list.append(self.pos_idx_mappings.get(pos, self.unk_pos_idx))
                head_idx_list.append(head)
            sentence_len = len(words_idx_list)

            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_head_list.append(torch.tensor(head_idx_list, dtype=torch.long, requires_grad=False))
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
    train = DpDataset(data_dir, str)
    train_dataloader = DataLoader(train, shuffle=True)
    print("")


if __name__ == '__main__':
    main()