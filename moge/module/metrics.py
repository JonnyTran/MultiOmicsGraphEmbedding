import torch


def top_k_multiclass(output: torch.Tensor, target: torch.Tensor, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    n_classes = target.size(1)

    _, top_indices = output.topk(topk, 1, True, True)
    print("top_indices", top_indices.shape, top_indices)

    # expanded_y = target.view(-1, 1).expand(-1, maxk)
    # print("expanded_y", expanded_y.shape, expanded_y)
    corrects = 0
    for i in range(0, batch_size):
        correct_percentage = target[i, top_indices[i]].float().sum(0) * 1.0 / topk
        print("correct_i", correct_percentage.shape, correct_percentage)
        corrects += correct_percentage

    return corrects.__mul__(1.0 / batch_size)
