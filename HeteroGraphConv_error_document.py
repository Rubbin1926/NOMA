import dgl
import torch
import dgl.nn.pytorch as dglnn
g = dgl.heterograph({
    ('job', 'to_job', 'job'): (torch.tensor([0]), torch.tensor([1])),
    ('job', 'to_machine', 'machine'): ([1, 1, 1, 1], [0, 1, 2, 3])})
print(g)


# case1:
# F.linear(input, self.weight, self.bias)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x3 and 1x2)
"""
conv = dglnn.HeteroGraphConv({
    'to_job' : dglnn.SAGEConv(3, 5, aggregator_type='mean'),
    'to_machine': dglnn.SAGEConv(1, 2, aggregator_type='mean')},
    aggregate='sum')

# mat1:起始节点有多少个 * 到这个起始节点的边的in_feats数量, mat2:边的in_feats数量 * 边的out_feats数量

h1 = {'job': torch.randn((g.number_of_nodes('job'), 3)),
      'machine': torch.randn((g.number_of_nodes('machine'), 10))}
h2 = conv(g, h1)
print(h2)
"""


# case2:
# F.linear(input, self.weight, self.bias)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x10 and 3x2)
"""
conv = dglnn.HeteroGraphConv({
    'to_job' : dglnn.SAGEConv(3, 5, aggregator_type='mean'),
    'to_machine': dglnn.SAGEConv(3, 2, aggregator_type='mean')},
    aggregate='sum')

# mat1:起始节点有多少个 * 目标节点的feature的数目, mat2:到这个起始节点的边的in_feats数量 * 边的out_feats数量

h1 = {'job': torch.randn((g.number_of_nodes('job'), 3)),
      'machine': torch.randn((g.number_of_nodes('machine'), 10))}
h2 = conv(g, h1)
print(h2)
"""
