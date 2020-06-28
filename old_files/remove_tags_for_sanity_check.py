file_to_write = 'restored_comp_file.labeled'
file_to_restore = 'tagged_comp_file.labeled'


with open(file_to_write, 'w') as file_writer:
    with open(file_to_restore, 'r') as file_reader:
        for i, line in enumerate(file_reader):
            if line.strip():
                split_words = line.split('\t')
                split_words[6] = '_'
                new_line = '\t'.join(split_words)
                file_writer.write(new_line)
                print(new_line)
            else:
                file_writer.write('\n')
