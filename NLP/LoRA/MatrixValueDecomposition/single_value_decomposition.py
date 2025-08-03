"""
Single value decompositoin is used in LoRA implementation.
Any matrix "A" with size (m x n) can be decomposed in [U . Sigma . V^T]
where 'U' : m x m matrix
Sigma : m x n diagonal matrix
V^T : transpose of n X n matrix.


@yota
"""

import torch
import torch.nn as nn
import numpy as np

A_rank = 2

A = torch.arange(start=0,end=10,dtype=torch.float).view(-1,2) #Rank of this matrix = 2
# print(A.shape)
U,S,V = torch.svd(A)

U_r = U[...,:A_rank]
S_r = torch.diag(S[:A_rank])
V_r = V[:,:A_rank].t()

B = U_r @ S_r

A = V_r


print(B)
print(A)

print(B@A)