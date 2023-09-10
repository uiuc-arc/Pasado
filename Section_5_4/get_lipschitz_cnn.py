import os
from timeit import default_timer as timer
import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

import sys

sys.path.insert(0, '../forward_mode_tensorized_src')

import argparse

from SimpleZono import *
from Duals import *
from precise_transformer import *
from cnn_helper import ConvInterval, ConvZonotope, ConvPasado

torch.set_default_dtype(torch.float64)
device = 'cpu'

parser = argparse.ArgumentParser(description='Get haze Lipschitz constant of MNIST CNN')
parser.add_argument('--net', choices=['small', 'med', 'big'], help='CNN architecture',
                    default='small')
parser.add_argument('-l', action='store_true', help='run the full experiment')
args = parser.parse_args()

num_images_to_test = 30 if args.l else 3

test_transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST(root='./MNIST_Data', train=False,
                                     download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

from model import Conv

net_type = args.net
print(f'======== Conv{net_type.capitalize()} ========')

if net_type == 'small':
    net = Conv(device, 28, [(16, 4, 2, 0), (32, 4, 2, 0)], [100, 10]).to(device)  # ConvSmall.
elif net_type == 'med':
    net = Conv(device, 28, [(16, 4, 2, 1), (32, 4, 2, 1)], [1000, 10]).to(device)  # ConvMed.
elif net_type == 'big':
    net = Conv(device, 28, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 3, 1, 1), (64, 4, 2, 1)], [512, 512, 10]).to(
        device)  # ConvBig.
else:
    raise ValueError("Undefined CNN architecture!")

net.load_state_dict(torch.load(f'trained/conv{net_type}.pth', map_location='cpu'))
net.eval()

correct_indices = torch.load(f'trained/indices_conv{net_type}.pth', map_location='cpu')

lc_zonos = []
lc_pasados = []
lc_intervals = []

time_zonos = []
time_intervals = []
time_pasados = []

n_splits_interval = 1
n_splits_zono = 1
n_splits_pasado = 1

init_shape = (1, 28, 28)

test_range = [10 ** (-k / 4) * 2 for k in range(2, 7)]
pbar_total = tqdm(total=len(test_range), position=0)

for epsilon in test_range:
    print(f'Running epsilon={epsilon}')
    lc_zono_total = 0
    lc_pasado_total = 0
    lc_interval_total = 0

    time_zono_total = 0
    time_interval_total = 0
    time_pasado_total = 0

    img_index = 0
    correct_images = 0

    with torch.no_grad(), tqdm(total=num_images_to_test, position=1) as pbar_eps:
        for (image, _) in testloader:
            if img_index in correct_indices:
                img_f = image.flatten()

                # Zonotope.
                start = timer()

                delta = epsilon / n_splits_zono
                interval_eps_l = []

                for i in range(n_splits_zono):
                    l_ = i * delta
                    r_ = l_ + delta
                    interval_eps = DualIntervalTensor(real_l=torch.tensor([l_]),
                                                      real_u=torch.tensor([r_]))
                    interval_eps.e1_l[0] = 1
                    interval_eps.e1_u[0] = 1
                    interval_eps_l.append(interval_eps)

                lc_zono_temp = []
                for i in range(n_splits_zono):
                    zono_eps = HyperDualIntervalToDualZonotope(interval_eps_l[i])
                    hazed_zono = (torch.tensor([1]) - zono_eps) * img_f + zono_eps
                    output = ConvZonotope(net, hazed_zono, init_shape)
                    lc_zono_ = torch.max(
                        torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))).item()
                    lc_zono_temp.append(lc_zono_)

                lc_zono = max(lc_zono_temp)
                end = timer()
                zono_time = end - start

                # Pasado
                start = timer()

                delta = epsilon / n_splits_pasado
                interval_eps_l = []

                for i in range(n_splits_pasado):
                    l_ = i * delta
                    r_ = l_ + delta
                    interval_eps = DualIntervalTensor(real_l=torch.tensor([l_]),
                                                      real_u=torch.tensor([r_]))

                    interval_eps.e1_l[0] = 1
                    interval_eps.e1_u[0] = 1
                    interval_eps_l.append(interval_eps)

                lc_pasado_temp = []
                for i in range(n_splits_pasado):
                    zono_eps = HyperDualIntervalToDualZonotope(interval_eps_l[i])
                    hazed_zono = (torch.tensor([1]) - zono_eps) * img_f + zono_eps
                    output = ConvPasado(net, hazed_zono, init_shape)
                    lc_pasado_ = torch.max(
                        torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))).item()
                    lc_pasado_temp.append(lc_pasado_)
                lc_pasado = max(lc_pasado_temp)
                end = timer()
                pasado_time = end - start

                # Interval.
                start = timer()
                lc_interval_temp = []
                for i in range(n_splits_interval):
                    hazed_int = (1 - interval_eps_l[i]) * img_f + interval_eps_l[i]
                    output = ConvInterval(net, hazed_int, init_shape)
                    lc_interval_ = torch.max(torch.maximum(torch.abs(output.e1_l), torch.abs(output.e1_u))).item()
                    lc_interval_temp.append(lc_interval_)
                lc_interval = max(lc_interval_temp)
                end = timer()
                interval_time = end - start

                time_zono_total += zono_time
                time_interval_total += interval_time
                time_pasado_total += pasado_time

                lc_zono_total += lc_zono
                lc_pasado_total += lc_pasado
                lc_interval_total += lc_interval

                correct_images += 1
                pbar_eps.update(1)

            if correct_images == num_images_to_test:
                break

            img_index += 1

    lc_zonos.append(lc_zono_total / correct_images)
    lc_pasados.append(lc_pasado_total / correct_images)
    lc_intervals.append(lc_interval_total / correct_images)

    time_zonos.append(time_zono_total / correct_images)
    time_intervals.append(time_interval_total / correct_images)
    time_pasados.append(time_pasado_total / correct_images)

    pbar_total.update(1)

pbar_total.close()
os.system('mkdir -p results')
torch.save(lc_zonos, f'results/lc_zonos_conv{net_type}.pth')
torch.save(lc_intervals, f'results/lc_intervals_conv{net_type}.pth')
torch.save(lc_pasados, f'results/lc_pasados_conv{net_type}.pth')

torch.save(time_zonos, f'results/time_zonos_conv{net_type}.pth')
torch.save(time_intervals, f'results/time_intervals_conv{net_type}.pth')
torch.save(time_pasados, f'results/time_pasados_conv{net_type}.pth')
