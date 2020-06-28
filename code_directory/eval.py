import torch
from code_directory.Models import AdvancedNet, BaseNet
from code_directory.inference import compute_uas


def load_model(model_path, model_type, return_indexing_dictionaries=True):
    saved_model = torch.load(model_path)
    if model_type == 'base':
        model = BaseNet(**saved_model['args'])
    if model_type == 'advanced':
        model = AdvancedNet(**saved_model['args'])
    model.load_state_dict(saved_model['state_dict'])
    model.eval()
    if return_indexing_dictionaries:
        return model, saved_model['indexing_dictionaries']
    return model


def eval_model(model, loader, loss=None, uas_list: list = None, loss_list: list = None):
    model.eval()
    num_sentences = len(loader)
    num_total = 0
    num_correct = 0
    total_loss = 0.
    for i, input_data in enumerate(loader):
        words_idx_tensor, pos_idx_tensor, true_heads, _ = input_data
        true_heads = true_heads.squeeze(0)
        num_total += true_heads.shape[0]
        scores = model(words_idx_tensor, pos_idx_tensor)
        _, curr_num_correct = compute_uas(scores, true_heads)
        num_correct += curr_num_correct
        if loss is not None:
            total_loss += loss(scores.to("cpu"), true_heads).item()/num_sentences
    uas = num_correct / num_total
    if uas_list is not None:
        uas_list.append(uas)
    if loss is not None:
        if loss_list is not None:
            loss_list.append(total_loss)
        return uas, total_loss
    return uas

