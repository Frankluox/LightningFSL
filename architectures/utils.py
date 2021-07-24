import torch
from torch import Tensor
def L2SquareDist(A: Tensor, B: Tensor, average: bool = True) -> Tensor:
    r"""calculate parwise euclidean distance between two batchs of features.

    Args:
        A: Torch feature tensor. size:[Batch_size, Na, nC]
        B: Torch feature tensor. size:[Batch_size, Nb, nC]
    Output:
        dist: The calculated distance tensor. size:[Batch_size, Na, Nb]
    """
    assert A.dim() == 3
    assert B.dim() == 3
    assert A.size(0) == B.size(0) and A.size(2) == B.size(2)
    nB = A.size(0)
    Na = A.size(1)
    Nb = B.size(1)
    nC = A.size(2)

    # AB = A * B = [nB x Na x nC] * [nB x nC x Nb] = [nB x Na x Nb]
    AB = torch.bmm(A, B.transpose(1, 2))

    AA = (A * A).sum(dim=2, keepdim=True).view(nB, Na, 1)  # [nB x Na x 1]
    BB = (B * B).sum(dim=2, keepdim=True).view(nB, 1, Nb)  # [nB x 1 x Nb]
    # l2squaredist = A*A + B*B - 2 * A * B
    dist = AA.expand_as(AB) + BB.expand_as(AB) - 2 * AB
    if average:
        dist = dist / nC

    return dist