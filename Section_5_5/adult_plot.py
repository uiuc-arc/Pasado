import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

colors = ["#1B9E77", "#D95F02", "#7570B3"]
markers = ["o", "^", "s"]  # circle, triangle, square
sns.set_style('darkgrid')

matplotlib.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
})

df = pd.read_csv('results/adult.csv', header=None)

epsilons = df[0].values
intervals = [int(t.split(",")[2].strip(")")) for t in df[1].values]
zonotopes = [int(t.split(",")[2].strip(")")) for t in df[2].values]
pasados = [int(t.split(",")[2].strip(")")) for t in df[3].values]

mask = epsilons <= 0.8
epsilons = epsilons[mask]
intervals = [intervals[i] for i, val in enumerate(mask) if val]
zonotopes = [zonotopes[i] for i, val in enumerate(mask) if val]
pasados = [pasados[i] for i, val in enumerate(mask) if val]

plt.figure(figsize=(8, 6))
plt.plot(epsilons, intervals, color=colors[0], marker=markers[0], label='Interval')
plt.plot(epsilons, zonotopes, color=colors[1], marker=markers[1], label='Zonotope')
plt.plot(epsilons, pasados, color=colors[2], marker=markers[2], label='Pasado')

plt.xlabel(r'$L_{\infty}$-ball radius $\epsilon$')
plt.ylabel(r'\# Verifiably Monotonic Features')
plt.legend()

plt.savefig('img/adult.jpg', bbox_inches='tight')
