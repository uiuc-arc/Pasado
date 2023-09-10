import torch

for domain in ['intervals', 'zonos', 'pasados']:
    print(f'{domain} runtimes:')

    small = torch.load(f'results/time_{domain}_convsmall.pth')
    med = torch.load(f'results/time_{domain}_convmed.pth')
    big = torch.load(f'results/time_{domain}_convbig.pth')

    print(f'ConvSmall: {sum(small) / len(small)}s')
    print(f'ConvMed: {sum(med) / len(med)}s')
    print(f'ConvBig: {sum(big) / len(big)}s')
