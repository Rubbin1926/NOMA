import dgl
import torch
import dgl.nn.pytorch as dglnn
g = dgl.heterograph({
    ('job', 'to_job', 'job'): (torch.tensor([0]), torch.tensor([1])),
    ('job', 'to_machine', 'machine'): ([1, 2, 3], [0, 1, 1]),
    ('machine', 'to_terminal', 'terminal'): ([0, 1], [0, 0])})
print(g)

conv = dglnn.HeteroGraphConv({
    'to_job': dglnn.GraphConv(5, 1),
    'to_machine': dglnn.GraphConv(5, 3),
    'to_terminal': dglnn.GraphConv(3, 2)},
    aggregate='sum')

h1 = {'job': torch.randn((g.number_of_nodes('job'), 5)),
      'machine': torch.randn((g.number_of_nodes('machine'), 3)),
      'terminal': torch.randn((g.number_of_nodes('terminal'), 1000))}
h2 = conv(g, h1)
print(h2)
