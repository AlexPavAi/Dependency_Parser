import os
import time

import torch
from torch.utils.data import DataLoader
import numpy as np

from code_directory.data_loader import DpDataset
from code_directory.eval import load_model
from code_directory.inference import infer_heads


def tag_file(dir_path: str, file: str, out_path, model_path, model_type, time_run=False):
    """
    :param out_path: the path of the output
    :param dir_path: the path of the directory of the file to tag
    :param file: the name of the file to tag
    :param model_path: the path of the model to tag with
    :param model_type: the model type 'advanced' or base
    :param time_run: if True times the run
    :return:
    """
    if time_run:
        t0 = time.time()
    model, indexing_dictionaries = load_model(model_path=model_path, model_type=model_type,
                                              return_indexing_dictionaries=True)
    dataset = DpDataset(dir_path, file.split('.')[0], indexing_dictionaries=indexing_dictionaries)
    loader = DataLoader(dataset, shuffle=False)
    num_sentences = len(loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inferred_head_all = np.zeros((num_sentences, 1), dtype='object')
    for i, input_data in enumerate(loader):
        words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
        true_heads = true_heads.squeeze(0)
        scores = model(words_idx_tensor, pos_idx_tensor)
        infered_heads = infer_heads(scores)
        inferred_head_all[i, 0] = infered_heads
        assert ((true_heads.shape[0]) == infered_heads.shape[0])
    file_to_tag = os.path.join(dir_path, file)
    file_to_write = out_path
    sentence_counter = 0
    word_in_sentence = 0
    with open(file_to_write, 'w') as file_writer:
        with open(file_to_tag, 'r') as file_reader:
            for i, line in enumerate(file_reader):
                sentence_tags = inferred_head_all[sentence_counter, 0]
                if line.strip():
                    split_words = line.split('\t')
                    infered_head = sentence_tags[word_in_sentence]
                    split_words[6] = str(infered_head)
                    new_line = '\t'.join(split_words)
                    file_writer.write(new_line)

                    word_in_sentence += 1
                else:
                    file_writer.write('\n')
                    sentence_counter += 1
                    word_in_sentence = 0
    if time_run:
        print('training took:', time.time()-t0)


if __name__ == '__main__':
    tag_file('data', 'test.labeled', 'tagged_test_file_m1.labeled', './basic_model.pkl', 'base',
             time_run=True)
    tag_file('data', 'test.labeled', 'tagged_test_file_m2.labeled', './advanced_model.pkl', 'advanced',
             time_run=True)
