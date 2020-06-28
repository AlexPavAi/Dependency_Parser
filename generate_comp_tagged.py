from code_directory.tag_file import tag_file


def generate_comp_tagged(model='both'):
    """

    :param model: the model to tag with base for the basic model advanced for the advanced model and both for both
    """
    if model == 'base' or model == 'both':
        tag_file(dir_path='./code_directory/data', file='comp.unlabeled', out_path='comp_m1_318556206.labeled',
                 model_path='./code_directory/basic_model.pkl', model_type='base')
    if model == 'advanced' or model == 'both':
        tag_file(dir_path='./code_directory/data', file='comp.unlabeled', out_path='comp_m2_318556206.labeled',
                 model_path='./code_directory/advanced_model.pkl', model_type='advanced')


if __name__ == '__main__':
    generate_comp_tagged()
