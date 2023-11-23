import dgl
import torch
import dgl.nn.pytorch as dglnn
g = dgl.heterograph({
    ('job', 'to_job', 'job'): (torch.tensor([0]), torch.tensor([1])),
    ('job', 'to_machine', 'machine'): ([1, 2, 3], [0, 1, 1]),
    ('machine', 'to_terminal', 'terminal'): ([0, 1], [0, 0])})
print(g)

conv = dglnn.HeteroGraphConv({
    'to_job': dglnn.SAGEConv(5, 1, aggregator_type='mean'),
    'to_machine': dglnn.SAGEConv(5, 2, aggregator_type='mean'),
    'to_terminal': dglnn.SAGEConv(5, 3, aggregator_type='pool')},
    aggregate='sum')

h1 = {'job': torch.randn((g.number_of_nodes('job'), 5)),
      'machine': torch.randn((g.number_of_nodes('machine'), 5)),
      'terminal': torch.randn((g.number_of_nodes('terminal'), 5))}
h2 = conv(g, h1)
print(h2)
