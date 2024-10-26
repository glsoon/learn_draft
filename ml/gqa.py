#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/26 21:29
# @Author  : Gongle
# @File    : gqa.py
# @Version : 1.0
# @Desc    : None


# shapes: (batch_size, seq_len, num_heads, head_dim)
query = torch.randn(1, 256, 8, 64)
key = torch.randn(1, 256, 2, 64)
value = torch.randn(1, 256, 2, 64)

num_head_groups = query.shape[2] // key.shape[2]
print(num_head_groups) # each group is of size 4 (群组类大小: group_size=4, 群组数量: group_nums=2) , since there are 2 kv_heads


## for the efficiency lets swap the seq_len and num_heads dimension
query = rearrange(query, "b n h d -> b h n d") ## torch.Size([1, 8, 256, 64])
key = rearrange(key, "b s h d -> b h s d") ## torch.Size([1, 2, 256, 64])
value = rearrange(value, "b s h d -> b h s d")

## original dimension 8 (number of heads for query) is now split in 2 groups
# (to match number of heads in key and value) of size 4 each.
## 2个组，一组4个头(组内大小为4)

# 上面query,key,value都已经转置
# 计算注意力相关度，所以query再转置，其余2个没有转置
query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)  ## 组内头数: num_head_groups=4
print(query.shape) # torch.Size([1, 4, 2, 256, 64])

# g stands for the size of groups
# h stands for the hidden dim
# n and s are equal and stands for sequence length
scores = einsum(query, key, "b g h n d, b h s d -> b h n s")
## (1,4,2,256,64) (1,2,256,64)  -> [1, 2, 256, 256]

print(scores.shape)  #torch.Size([1, 2, 256, 256])

"""
Behind the scenes einsum did two things:

1)It made a matrix multiplication of query and key . 
In our example those tensors were of shape (1, 4, 2, 256, 64) and (1, 2, 256, 64), 
so matrix multiplication along last two dimensions gives (1, 4, 2, 256, 256).

2)Now we need to sum elements across the second dimension (dimension g ) — 
This is what einsum will do for us automatically if the dimension is omitted in the specified output shape. 
We need to make such a summation to match the number of heads in keys and values.

einsum(query, key, "b g h n d, b h s d -> b h n s")
操作操作包含2个：
1. (1, 4, 2, 256, 64)@(1, 2, 256, 64)根据最后2维得到 (1, 4, 2, 256, 256).
2. (1, 4, 2, 256, 256)由于结果g维度消失，所以根据g维度自动求和，得到最终[1, 2, 256, 256]，以至于匹配keys和values。

"""

import torch.nn.functional as F

scale = query.size(-1) ** 0.5
attention = F.softmax(scores / scale, dim=-1)

# here we do just a standard matrix multiplication
out = einsum(attention, value, "b h n s, b h s d -> b h n d")

# finally, just reshape back to the (batch_size, seq_len, num_kv_heads, hidden_dim)
out = rearrange(out, "b h n d -> b n h d")
print(out.shape) # torch.Size([1, 256, 2, 64])




