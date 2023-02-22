from densenet import DenseNetFCN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input = torch.rand((1, 1, 64, 64, 64)).to(device)

# settings for DenseNet103, which are default
model = DenseNetFCN(
    ch_in=1,
    ch_out_init=48,
    num_classes=2,
    growth_rate=16,
    layers=(4, 5, 7, 10, 12),
    bottleneck=True,
    bottleneck_layer=15
).to(device)

output = model(input)


'''
OUTPUT:
down block 1 exit shape: torch.Size([1, 112, 64, 64, 64])
down block 2 exit shape: torch.Size([1, 192, 32, 32, 32])
down block 3 exit shape: torch.Size([1, 304, 16, 16, 16])
down block 4 exit shape: torch.Size([1, 464, 8, 8, 8])
down block 5 exit shape: torch.Size([1, 656, 4, 4, 4])
bottleneck exit shape: torch.Size([1, 896, 2, 2, 2])
up block 1 exit shape: torch.Size([1, 1088, 4, 4, 4])
up block 2 exit shape: torch.Size([1, 816, 8, 8, 8])
up block 3 exit shape: torch.Size([1, 576, 16, 16, 16])
up block 4 exit shape: torch.Size([1, 384, 32, 32, 32])
up block 5 exit shape: torch.Size([1, 256, 64, 64, 64])
'''