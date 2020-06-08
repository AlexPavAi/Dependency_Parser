from chu_liu_edmonds import decode_mst
import torch
import numpy as np


def infer_heads(scores, squeeze=True):
    """
    infer the heads
    :param scores: a tensor from the shape (n+1, n) (or (1, n+1, n) if squeeze==True where n is the length of the
    sentence) such that scores[h, m-1] is the
    score of (h, m)
    :param squeeze: if the input is from the shape (1, n+1, n) (as the network outputs) the first dimension will be
    squeezed
    :return: np array of inferred heads where the first value is the head of the first word of the sentence and so on
    """
    if squeeze:
        scores = torch.squeeze(scores, 0)
    length = scores.shape[0]
    weights = np.empty((length, length))
    weights[:, 1:] = scores.detach().cpu().numpy()
    weights[:, :1] = float('-inf')
    return decode_mst(weights, length, has_labels=False)[0][1:]


def compute_uas(scores, true_heads, squeeze=True):
    """

    :param scores: a tensor from the shape (n+1, n) (or (1, n+1, n) if squeeze==True where n is the length of the
    sentence) such that scores[h, m-1] is the
    score of (h, m)
    :param squeeze: if the input is from the shape (1, n+1, n) (as the network outputs) the first dimension will be
    squeezed
    :param true_heads: the true heads
    :return: the UAS and the number of correct dependencies (in that order)
    """
    true_heads = true_heads.numpy()
    inferred_heads = infer_heads(scores, squeeze=squeeze)
    num_correct = np.sum(true_heads == inferred_heads)
    return num_correct/len(true_heads), num_correct


def test_inference():
    weights = {(0, 1): 9,
         (0, 2): 10,
         (0, 3): 9,
         (1, 2): 20,
         (1, 3): 3,
         (2, 1): 30,
         (2, 3): 30,
         (3, 1): 11,
         (3, 2): 0}
    scores = torch.empty((1, 4, 3))
    for (i, j), w in weights.items():
        scores[0][i][j-1] = w
    print(compute_uas(scores, torch.tensor([2, 0, 2])))

