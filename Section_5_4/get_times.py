import torch

for domain in ['intervals', 'zonos', 'precise']:
    print(f'{domain} runtimes:')

    l3 = torch.load(f'results/time_{domain}_3layer.pth')
    l4 = torch.load(f'results/time_{domain}_4layer.pth')
    l5 = torch.load(f'results/time_{domain}_5layer.pth')
    lbig = torch.load(f'results/time_{domain}_big.pth')
    print(f'3layer: {sum(l3) / len(l3)}s')
    print(f'4layer: {sum(l4) / len(l4)}s')
    print(f'5layer: {sum(l5) / len(l5)}s')
    print(f'big: {sum(lbig) / len(lbig)}s')
    print()
