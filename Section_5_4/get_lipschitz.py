import sys

sys.path.insert(0, '../forward_mode_tensorized_src')

from SimpleZono import *
from Duals import *
from precise_transformer import *

import os
from timeit import default_timer as timer
import torch
import torchvision
import torchvision.transforms as transforms

torch.set_default_dtype(torch.float64)

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Get haze Lipschitz constant of MNIST Network')
parser.add_argument('--network', choices=['3layer', '4layer', '5layer', 'big'], help='neural network architecture',
                    default='3layer')
args = parser.parse_args()

num_images_to_test = 30

test_transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST(root='./MNIST_Data', train=False,
                                     download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

from model import FCN, FCNBig

network = args.network
print(f'===== {network} Network =====')

if network == '3layer':
    net = FCN(3)
    layers = 3
elif network == '4layer':
    net = FCN(4)
    layers = 4
elif network == '5layer':
    net = FCN(5)
    layers = 5
else:
    net = FCNBig()
net.load_state_dict(torch.load(f'trained/model_{network}.pth', map_location='cpu'))
net.eval()

if 'big' == network:  # define forward functions for the "big" network
    def forward_zono(x):
        x = SigmoidDualZonotope(AffineDualZonotope(x, net[1].weight.T) + net[1].bias)
        x = SigmoidDualZonotope(AffineDualZonotope(x, net[3].weight.T) + net[3].bias)
        x = SigmoidDualZonotope(AffineDualZonotope(x, net[5].weight.T) + net[5].bias)
        x = SigmoidDualZonotope(AffineDualZonotope(x, net[7].weight.T) + net[7].bias)
        x = AffineDualZonotope(x, net[9].weight.T) + net[9].bias
        return x


    def forward_interval(x):
        x = Sigmoid_di(x @ abstract_di(net[1].weight.T) + abstract_di(net[1].bias))
        x = Sigmoid_di(x @ abstract_di(net[3].weight.T) + abstract_di(net[3].bias))
        x = Sigmoid_di(x @ abstract_di(net[5].weight.T) + abstract_di(net[5].bias))
        x = Sigmoid_di(x @ abstract_di(net[7].weight.T) + abstract_di(net[7].bias))
        x = x @ abstract_di(net[9].weight.T) + abstract_di(net[9].bias)
        return x


    def forward_zono_precise(x):
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net[1].weight.T) + net[1].bias)
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net[3].weight.T) + net[3].bias)
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net[5].weight.T) + net[5].bias)
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net[7].weight.T) + net[7].bias)
        x = AffineDualZonotope(x, net[9].weight.T) + net[9].bias
        return x

else:  # define forward functions for the "3/4/5-layer" networks
    def forward_zono(x):
        x = SigmoidDualZonotope(AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
        x = SigmoidDualZonotope(AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
        if layers >= 4:
            x = SigmoidDualZonotope(AffineDualZonotope(x, net.fc3.weight.T) + net.fc3.bias)
        if layers == 5:
            x = SigmoidDualZonotope(AffineDualZonotope(x, net.fc4.weight.T) + net.fc4.bias.data)
        x = AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data
        return x


    def forward_zono_precise(x):
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
        x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
        if layers >= 4:
            x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc3.weight.T) + net.fc3.bias.data)
        if layers == 5:
            x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc4.weight.T) + net.fc4.bias.data)
        x = AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data
        return x


    def forward_interval(x):
        x = Sigmoid_di(x @ abstract_di(net.fc1.weight.T) + abstract_di(net.fc1.bias.data))
        x = Sigmoid_di(x @ abstract_di(net.fc2.weight.T) + abstract_di(net.fc2.bias.data))
        if layers >= 4:
            x = Sigmoid_di(x @ abstract_di(net.fc3.weight.T) + abstract_di(net.fc3.bias.data))
        if layers == 5:
            x = Sigmoid_di(x @ abstract_di(net.fc4.weight.T) + abstract_di(net.fc4.bias.data))
        x = x @ abstract_di(net.fc_final.weight.T) + abstract_di(net.fc_final.bias.data)
        return x

correct_indices = torch.load(f'trained/indices_{network}.pth', map_location=torch.device('cpu'))

lc_zonos = []
lc_zonos_precise = []
lc_intervals = []
time_zonos = []
time_intervals = []
time_precise = []

n_splits = 1

test_range = [10 ** (-k / 4) * 2 for k in range(2, 18)]
pbar_total = tqdm(total=len(test_range), position=0)

for epsilon in test_range:  # change back to 2,18
    print(f'Running epsilon={epsilon}')
    lc_zono_total = 0
    lc_zono_precise_total = 0
    lc_interval_total = 0

    time_zono_total = 0
    time_interval_total = 0
    time_precise_total = 0

    img_index = 0
    correct_images = 0

    delta = epsilon / n_splits
    interval_eps_l = []

    for i in range(n_splits):
        l_ = i * delta
        r_ = l_ + delta
        interval_eps = DualIntervalTensor(real_l=torch.tensor([l_]),
                                          real_u=torch.tensor([r_]))
        interval_eps.e1_l[0] = 1
        interval_eps.e1_u[0] = 1
        interval_eps_l.append(interval_eps)

    with torch.no_grad(), tqdm(total=num_images_to_test, position=1) as pbar_eps:
        for (image, _) in testloader:
            if img_index in correct_indices:
                img_f = image.flatten()

                # zonotope
                start = timer()
                lc_zono_temp = []
                for i in range(n_splits):
                    zono_eps = HyperDualIntervalToDualZonotope(interval_eps_l[i])
                    hazed_zono = (torch.tensor([1]) - zono_eps) * img_f + zono_eps
                    output = forward_zono(hazed_zono)
                    lc_zono_ = torch.max(
                        torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))).item()
                    lc_zono_temp.append(lc_zono_)

                lc_zono = max(lc_zono_temp)
                end = timer()
                zono_time = end - start

                # PRECISE Zonotope
                start = timer()
                lc_zono_precise_temp = []
                for i in range(n_splits):
                    zono_eps = HyperDualIntervalToDualZonotope(interval_eps_l[i])
                    hazed_zono = (torch.tensor([1]) - zono_eps) * img_f + zono_eps
                    output = forward_zono_precise(hazed_zono)
                    lc_zono_precise_ = torch.max(
                        torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))).item()
                    lc_zono_precise_temp.append(lc_zono_precise_)
                lc_zono_precise = max(lc_zono_precise_temp)
                end = timer()
                precise_zono_time = end - start

                # interval
                start = timer()
                lc_interval_temp = []
                for i in range(n_splits):
                    hazed_int = (1 - interval_eps_l[i]) * img_f + interval_eps_l[i]
                    output = forward_interval(hazed_int)
                    lc_interval_ = torch.max(torch.maximum(torch.abs(output.e1_l), torch.abs(output.e1_u))).item()
                    lc_interval_temp.append(lc_interval_)
                lc_interval = max(lc_interval_temp)
                end = timer()
                interval_time = end - start

                time_zono_total += zono_time
                time_interval_total += interval_time
                time_precise_total += precise_zono_time

                lc_zono_total += lc_zono
                lc_zono_precise_total += lc_zono_precise
                lc_interval_total += lc_interval

                correct_images += 1
                pbar_eps.update(1)

            if correct_images == num_images_to_test:
                break

            img_index += 1

    lc_zonos.append(lc_zono_total / correct_images)
    lc_zonos_precise.append(lc_zono_precise_total / correct_images)
    lc_intervals.append(lc_interval_total / correct_images)

    time_zonos.append(time_zono_total / correct_images)
    time_intervals.append(time_interval_total / correct_images)
    time_precise.append(time_precise_total / correct_images)

    pbar_total.update(1)

pbar_total.close()
os.system('mkdir -p results')
torch.save(lc_zonos, f'results/lc_zonos_{network}.pth')
torch.save(lc_intervals, f'results/lc_intervals_{network}.pth')
torch.save(lc_zonos_precise, f'results/lc_precise_{network}.pth')

torch.save(time_zonos, f'results/time_zonos_{network}.pth')
torch.save(time_intervals, f'results/time_intervals_{network}.pth')
torch.save(time_precise, f'results/time_precise_{network}.pth')
