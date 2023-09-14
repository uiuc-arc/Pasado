# Pasado

This repository contains the implementation for the paper "Synthesizing Precise Static Analyzers for Automatic
Differentiation" (OOPSLA 2023)
by Jacob Laurel, Siyuan Brant Qian, Gagandeep Singh, Sasa Misailovic. In this paper, we present Pasado, a technique for
synthesizing precise static analyzers for Automatic Differentiation. Our technique allows one to automatically construct
a static analyzer specialized for the Chain Rule, Product Rule, and Quotient Rule computations for Automatic
Differentiation in a way that abstracts all the nonlinear operations of each respective rule simultaneously.

Please find the most updated version of our artifact at [this GitHub repository](https://github.com/uiuc-arc/Pasado).

## Quick Navigation

- [Code Documentation](#code-documentation)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Section 5.2 & Example](#validating-section-52-experiments-and-example)
- [Section 5.3](#validating-section-53-experiments)
- [Section 5.4](#validating-section-54-experiments)
- [Section 5.5](#validating-section-55-experiments)

## Code Documentation

A complete description of the source files and their functionality is provided in a separate
document [here](https://github.com/uiuc-arc/Pasado/blob/main/Code_Documentation.pdf).

## Requirements

This artifact is built on the following requirements:

| Library              | Required Version | Possibly Compatible Versions | Notes                      |
|----------------------|------------------|------------------------------|----------------------------|
| **Python**           | 3.10.6           | 3.9 or greater               |                            |
| **Numpy**            | 1.25.1           | 1.19.5 or greater            |                            |
| **Affapy**           | 0.1              |                              |                            |
| **Scikit-learn**     | 1.3.0            | 0.24.2 or greater            |                            |
| **Seaborn**          | 0.12.2           | 0.11.2 or greater            |                            |
| **MatPlotLib**       | 3.7.2            | 3.7.1 or greater             |                            |
| **tabulate**         | 0.9.0            |                              |                            |
| **tqdm**             | 4.65.0           |                              |                            |
| **PyTorch**          | 2.0.1+cpu        | 1.9.0 or greater             | Should also work with GPU. |
| **Torchaudio**       | 2.0.2+cpu        | 0.13.1 or greater            |                            |
| **Torchvision**      | 0.15.2+cpu       | 0.14.1 or greater            |                            |
| **Jupyter Notebook** | 6.5.4            | 5.7.11 or greater            |                            |
| **IPython**          | 8.14.0           | 5.8.0 or greater             |                            |

## Directory Structure

Inside the main directory `Pasado` are the following
subdirectories: `Section_5_2`, `Section_5_3`,  `Section_5_4`, `Section_5_5`, `forward_mode_non_tensorized_src`, `forward_mode_tensorized_src`, `reverse_mode_non_tensorized_src`.
The latter three directories all correspond to the source code, which has been separated from the experimental
evaluation code. The experimental evaluation code is in the `Section_5_2`, `Section_5_3`, `Section_5_4`,
and `Section_5_5` subdirectories.

## Getting Started

To get started, please clone this repository from GitHub and move into its directory by running the following commands
in your terminal:

```bash
git clone https://github.com/uiuc-arc/Pasado.git
cd Pasado
```

The instructions that follow assume that the reader is now in the `Pasado` directory.

## Validating Section 5.2 Experiments and Example

To validate the robust sensitivity analysis of ODEs experiments from Section 5.2 of the paper, we will enter into
the `Section_5_2` subdirectory. This can be done by running the following command:

```bash
cd Section_5_2
```

Inside this directory are the experiments for performing robust AD-based sensitivity analysis of ODE solutions. The two
main scripts are the `climate.sh` and `chemical.sh`. While the `climate.sh` script is relatively quick to
run, `chemical.sh` will take much longer.

To run the experiments for the climate model run the `climate.sh` file using the following command:

```bash
./climate.sh
```

The results can be visualized by going into the `img` directory with the following command:

```bash
cd Section_5_2/img
```

To view these figures we recommend the following commands:

```bash
eog climate_step.jpg  # corresponds to Fig 3
eog climate_scatter.jpg  # corresponds to Fig. 4b
``` 

Where climate_step.jpg corresponds to Fig 3 of the paper (in the Example section) and climate_scatter.jpg corresponds to
Fig. 4b of the paper.

To validate the Chemical ODE robust AD-based sensitivity analysis, we now exit the `img` directory and return back to
the `Section_5_2` directory using the following command:

```bash
cd Section_5_2
```

And then to actually run the Chemical ODE experiment, we run the following command:

```bash
./chemical.sh
```

As noted, this experiment will take a while to run, hence we only present results for a subset of the experiments by
default. To run the whole experiment and get Fig. 4a, one would add the optional `-l` flag and thus instead run the
following command:

```bash
./chemical.sh -l
```

However this option is **VERY slow** (will take a few hours) so we **highly** recommend running `./chemical.sh` instead
of `./chemical.sh -l`.

The results can be visualized by again going into the `img` directory with the following command:

```bash
cd Section_5_2/img
```

To view the figures for the chemical ODE experimental results, we recommend the following commands:

```bash
eog chemical_step.jpg  # corresponds to Fig. 5
eog chemical_scatter.jpg  # corresponds to (a subset of) Fig. 4a
``` 

Where `chemical_step.jpg` shows the results of Fig. 5 and `chemical_scatter.jpg` shows the results of Fig 4a. Note that
if one ran the shorter command `./chemical.sh` instead of the longer-running `./chemical.sh -l`, then
the `chemical_scatter.jpg` plot will only have **a subset** of the results of Fig. 4a, because only a subset of the
total experiments were run. The NeuralODE's network is trained in the script we provide so that our entire workflow is
visible to the reader, however, due to the internal behaviour of sklearn, the training process may lead to slightly
different networks on different machines (due to internal floating-point differences), thus the results may look
slightly different compared to those in the submitted paper, as the analysis is being applied to slightly different
networks. However, in all cases the trend of Pasado being strictly more precise than zonotope AD for these benchmarks
always holds true.

Furthermore, the same information presented in a tabular form can be found in the `data` subdirectory and can be
accessed via the following command:

```bash
cd Section_5_2/data
```

The tabular form of the same result is presented in both CSV and HTML formats. We recommend viewing the results in the
HTML format (using a browser such as firefox). This can be done using the following commands:

```bash
firefox chemical_ode_table.html
firefox climate_ode_table.html
```

To clear all the experimental results (if one wishes to do a fresh rerun, perhaps after modifying some part of the
code), we simply run the following command:

```bash
./clear.sh
```

### Source Code for Section 5.2 Experiments

The source code that implements the abstractions that are evaluated in the Section 5.2 experiments is found in
the `forward_mode_non_tensorized_src` directory. This naming convention follows from the fact that this code is
forward-mode AD, but the implementation of the abstraction (specifically the zonotopes) is not tensorized, since unlike
standard DNNs, these ODE solver computations do not satisfy a tensorized structure and thus one cannot vectorize the
zonotope implementation. This directory can be accessed via the following command:

```bash
cd forward_mode_non_tensorized_src
```

A description of the source files and their functionality is provided in a separate
document [here](https://github.com/uiuc-arc/Pasado/blob/main/Code_Documentation.pdf).

## Validating Section 5.3 Experiments

Here, we detail the steps needed to reproduce our Black-Scholes experimental results in Section 5.3 of the paper. We
will first enter into the `Section_5_3` subdirectory. This can be done by running the following command:

```bash
cd Section_5_3
```

To run the experiments for the Black-Scholes model, run the `black_scholes.sh` file using the following command:

```bash
./black_scholes.sh
```

The results can be visualized by going into the `img` directory with the following command:

```bash
cd Section_5_3/img
```

To view the results we recommend the following command:

```bash
eog black_scholes_rev.jpg  # corresponds to Fig. 6
``` 

This plot shows the results as described in Fig. 6 of the paper.

The same information presented in a tabular form can be found in the `data` subdirectory which can be accessed via the
following command:

```bash
cd Section_5_3/data
```

The tabular form of the same result is presented in both CSV and HTML formats. We recommend viewing the results in the
HTML format (using a browser such as Firefox). This can be done using the following commands:

```bash
firefox black_scholes_rev_K.html
firefox black_scholes_rev_r.html
firefox black_scholes_rev_S.html
firefox black_scholes_rev_tau.html
firefox black_scholes_rev_sigma.html
```

To clear all the experimental results (if one wishes to do a fresh rerun, perhaps after modifying some part of the
code), we simply run the following command:

```bash
./clear.sh
```

### Source Code for Section 5.3 Experiments

The source code that implements the abstractions that are evaluated in the Section 5.3 experiments is found in
the `reverse_mode_non_tensorized_src` directory.

```bash
cd reverse_mode_non_tensorized_src
```

This naming convention follows from the fact that this code is reverse-mode AD, but the implementation of the
abstraction (specifically the zonotopes) is not tensorized, since unlike standard DNNs, the Black-Scholes computation
does not satisfy a tensorized structure and thus one cannot vectorize the zonotope implementation. This code is
reverse-mode since the Black-Scholes model has several inputs, but only a single output, meaning reverse-mode is more
efficient in this case than forward mode.

A complete description of the source files and their functionality is provided in a separate
document [here](https://github.com/uiuc-arc/Pasado/blob/main/Code_Documentation.pdf).

## Validating Section 5.4 Experiments

Here, we detail the steps needed to reproduce our local Lipschitz robustness experimental results in Section 5.4 of the
paper. We will first enter into the `Section_5_4` subdirectory. This can be done by running the following command

```bash
cd Section_5_4
```

Inside this subdirectory are the folders containing the trained networks, as well as the MNIST image data and lastly,
the scripts needed to run the experiments and plot the results.

The full FFNN experiments can be run via the following command:

```bash
./lipschitz.sh
```

It is important to note that in order to have the experiments run for a manageable amount of time for the artifact
evaluator (e.g. 5 minutes instead of 8+ hrs), we have cut down on the number of images we average over (compared to the
paper), thus the results may look slightly different from in the paper. However, this difference is miniscule.
Additionally, if one wishes to only run a subset of the FFNN experiments, such as for just a single one of the networks,
this can be done via:

```bash
python3 get_lipschitz.py --network <network name>
```

e.g., to certify the 3-layer network, run:

```bash
python3 get_lipschitz.py --network 3layer
```

However, we recommend running the `./get_lipschitz.sh` command as it will generate all the FFNN results. In addition
to running all the Lipschitz experiments,  `./get_lipschitz.sh` collects the results so that they can be plotted (
more detail given below).

The full CNN experiments can be run via the following command:

```bash
./lipschitz_cnn.sh
```

Note that in order to have the experiments run for a manageable amount of time for the artifact evaluator (e.g. 10
minutes instead of 24+ hrs), we have cut down on the number of images we average over (compared to the paper), thus the
results may look slightly different than in the paper. However, this difference is miniscule. Since the `ConvBig`
experiment consumes too much RAM (more than 64 GB), we have excluded it by default. To run the complete experiment (
including the `ConvBig` experiments), please run the following command:

```bash
./lipschitz_cnn.sh -l
```

Additionally, if one wishes to only run a subset of the CNN experiments, such as for just a single one of the networks,
this can be done via:

```bash
python3 get_lipschitz_cnn.py --net <network name>
```

e.g., to certify the ConvSmall network, run:

```bash
python3 get_lipschitz_cnn.py --net small
```

### Plotting

Upon completing the experiments, the results can be visualized through two Jupyter notebooks, which is accessible
through the following commands:

```bash
jupyter notebook
```

And then upon opening the jupyter notebook application, selecting the `Plot.ipynb` notebook for FFNN experiments
and `Plot_cnn.ipynb` notebook for CNN experiments. Simply press Shift+Enter to run each cell, and the plots will be
visualized.

### Source Code for Section 5.4 Experiments

The source code that implements the abstractions used in Section 5.4 is given in the `forward_mode_tensorized_src`
subdirectory which is accessible with the following command:

```bash
cd forward_mode_tensorized_src
```

Because neural networks have very specific structure (unlike general computations like in Black-Scholes or ODE solvers),
abstractly interpreting neural networks can be implemented in a vectorized manner. Hence why for these large DNN
benchmarks, we offer this additional implementation of our abstraction which uses tensor-level operations (such as for
the linear regression or root solving that Pasado performs) that is tailored to the structure of DNNs.

A full description of the source code is provided in separate
documentation [here](https://github.com/uiuc-arc/Pasado/blob/main/Code_Documentation.pdf).

### Exploring the Network Architectures

This part is completely optional, however for the interested reader who wishes to explore the network architectures we
used, the class definitions of the models can be found in `Section_5_4/model.py`. Saved models
parameters (`model_*.pth`) can be found in the `trained` subdirectory, which is accessible via the following command:

```bash
cd Section_5_4/trained
```

In the paper, we trained three FFNN networks with 3, 4, and 5 layers, with 100 neurons in each hidden layer, and a large
FFNN network with 5 layers, with 1024 neurons in each hidden layer. We name these
networks: `3layer`, `4layer`, `5layer`, and `big`. For CNN benchmarks, we trained three CNN networks as cited in the
paper, namely `ConvSmall`, `ConvMed`, and `ConvBig`. These networks can also be trained from scratch, but may take some
time and is not necessary, because as mentioned the pre-trained versions used in the paper are already provided in
the `trained` directory.

## Validating Section 5.5 Experiments

Here, we detail the steps needed to reproduce our Adult Income experimental results in Section 5.5 of the paper. We will
first enter into the `Section_5_5` subdirectory. This can be done by running the following command:

```bash
cd Section_5_5
```

To run the experiments for the Adult Income model, run the `adult.sh` file using the following command:

```bash
./adult.sh
```

The results can be visualized by going into the `img` directory with the following command:

```bash
cd Section_5_5/img
```

To view the results we recommend the following command:

```bash
eog adult.jpg  # corresponds to Fig. 10
``` 

This plot shows the results as described in Fig. 10 of the paper.

The same information presented in a tabular form can be found in the `data` subdirectory which can be accessed via the
following command:

```bash
cd Section_5_5/data
```

The tabular form of the same result is presented in both CSV and HTML formats. We recommend viewing the results in the
HTML format (using a browser such as Firefox). This can be done using the following command:

```bash
firefox adult.html
```

To clear all the experimental results (if one wishes to do a fresh rerun, perhaps after modifying some part of the
code), we simply run the following command:

```bash
./clear.sh
```

### Source Code for Section 5.5 Experiments

The source code that implements the abstractions that are evaluated in the Section 5.5 experiments is found in
the `reverse_mode_non_tensorized_src` directory.

```bash
cd reverse_mode_non_tensorized_src
```

This naming convention follows from the fact that this code is reverse-mode AD, but the implementation of the
abstraction (specifically the zonotopes) is not tensorized. This code is reverse-mode since the Adult Income model has
several inputs, but only a single output, meaning reverse mode is more efficient in this case than forward mode.

A complete description of the source files and their functionality is provided in a separate
document [here](https://github.com/uiuc-arc/Pasado/blob/main/Code_Documentation.pdf).


