import os
from misc import save_obj, load_obj

data_folder = 'data'
data_comp_file = 'comp.unlabeled'
data_train_file = 'train.labeled'
file_to_write = 'tagged_train_file.labeled'
inferred_head_table = load_obj('inferred_heads_train')
file_to_tag = os.path.join(data_folder,data_train_file)

sentence_counter = 0
word_in_sentence = 0
with open(file_to_write, 'w') as file_writer:
    with open(file_to_tag, 'r') as file_reader:
        for i, line in enumerate(file_reader):
            sentence_tags = inferred_head_table[sentence_counter, 0]
            if line.strip():
                split_words = line.split('\t')
                infered_head = sentence_tags[word_in_sentence]
                split_words[6] = str(infered_head)
                new_line = '\t'.join(split_words)
                file_writer.write(new_line)
                print(new_line)

                word_in_sentence += 1
            else:
                file_writer.write('\n')
                sentence_counter += 1
                word_in_sentence = 0
