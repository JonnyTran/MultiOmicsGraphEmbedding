import torch


def top_k_multiclass(output: torch.Tensor, target: torch.Tensor, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, top_indices = output.topk(maxk, 1, True, True)
    top_indices = top_indices.t()

    print("top_indices", top_indices.shape)

    classes_indices = torch.nonzero(target)
    print("classes_indices", classes_indices.shape)
    correct = top_indices.eq(classes_indices.view(1, -1).expand_as(top_indices))
    print("correct", correct.shape)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.__mul__(100.0 / batch_size))
    return res
