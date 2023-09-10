import torch
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
import itertools
from torch.nn.functional import softplus
from conv import torch_conv_layer_to_affine


def copy_centers(centers, num):
    cols = np.prod(list(centers.shape))
    return centers.expand((num, cols))


# converts row vectors like [a,b,c] to [[a,0,0],[0,b,0],[0,0,c]]
def traceify(rowvec):
    leng = torch.numel(rowvec)
    copied = copy_centers(rowvec, leng)
    identity = torch.eye(leng)
    return (identity * copied)


def IntervalsToZonotope(lower, upper):
    assert (lower.shape == upper.shape)
    assert (torch.all(lower <= upper))
    midpoint = (upper + lower) / 2.
    radii = (upper - lower) / 2.
    generators = traceify(radii)
    return Zonotope(midpoint, generators)


def ZonotopeToInterval(zonotope):
    return (zonotope.get_lb(), zonotope.get_ub())


class Zonotope:

    def __init__(self, centers, coefs):
        # assert(np.prod(list(centers.shape))==coefs.shape[1])
        self.centers = centers  # will be a 1 x M (where M is number of vars) row vector
        self.generators = coefs  # will be a N x M (where N is number of noise erms) matrix

    def clone(self):
        new_centers = self.centers.detach().clone()
        new_generators = self.generators.detach().clone()
        return Zonotope(new_centers, new_generators)

    def get_num_vars(self):
        return np.prod(list(self.centers.shape))

    def get_num_noise_symbs(self):
        return self.generators.shape[0]

    def get_coeff_abs(self):
        return torch.sum(torch.abs(self.generators), dim=0)  # sum along rows

    # tested - works
    def get_lb(self):
        cof_abs = self.get_coeff_abs()
        lb = self.centers - cof_abs
        return lb

    # tested - works
    def get_ub(self):
        cof_abs = self.get_coeff_abs()
        ub = self.centers + cof_abs
        return ub

    # tested - works
    def expand(self, n):
        if n == 0:
            return
        if n > 0:
            cols = self.generators.shape[1]
            self.generators = torch.cat([self.generators, torch.zeros((n, cols))], dim=0)
        else:
            raise Exception

    # tested - works
    def __add__(self, other):
        if type(other) in [float, int]:
            centers = self.centers + other
            return Zonotope(centers, self.generators)
        elif type(other) is Zonotope:
            # make sure same dimensions
            self_noise_syms = self.get_num_noise_symbs()
            other_noise_syms = other.get_num_noise_symbs()

            if self_noise_syms > other_noise_syms:
                other.expand(self_noise_syms - other_noise_syms)
            elif self_noise_syms < other_noise_syms:
                self.expand(other_noise_syms - self_noise_syms)

            centers = self.centers + other.centers  # add tensors elementwise
            generators = self.generators + other.generators  # add tensors elementwise
            return Zonotope(centers, generators)

        # tested - works
        # instead of adding the same constant to all centers, adds a different constant to each center
        elif type(other) in [torch.Tensor]:
            if (other.shape == self.centers.shape):
                centers = self.centers + other
                return Zonotope(centers, self.generators)
            else:
                raise Exception
        else:
            raise Exception

    __radd__ = __add__

    # tested - works
    def __str__(self):
        return ("Centers: \n" + self.centers.__str__() + "\n" + "Generators: \n" + self.generators.__str__() + "\n")

    # tested - works
    def __neg__(self):
        return Zonotope(-self.centers, self.generators)

        # tested - works

    def __sub__(self, other):
        return self + (-other)

    # tested - works
    def __mul__(self, other):
        if type(other) in [float, int]:  # scalar multiplication can be done exactly
            centers = other * self.centers
            generators = other * self.generators
            return Zonotope(centers, generators)

        elif type(other) is Zonotope:
            # make sure same dimensions
            self_noise_syms = self.get_num_noise_symbs()
            other_noise_syms = other.get_num_noise_symbs()

            if self_noise_syms > other_noise_syms:
                other.expand(self_noise_syms - other_noise_syms)
            elif self_noise_syms < other_noise_syms:
                self.expand(other_noise_syms - self_noise_syms)

            centers = self.centers * other.centers
            a0_copy = copy_centers(self.centers, self.get_num_noise_symbs())
            b0_copy = copy_centers(other.centers, other.get_num_noise_symbs())

            a0bi = a0_copy * other.generators
            b0ai = b0_copy * self.generators
            before_adding_new_terms = a0bi + b0ai

            abs_ai = self.get_coeff_abs()
            abs_bi = other.get_coeff_abs()
            new_noise_magnitudes = abs_ai * abs_bi

            # need to convert new noise magnitudes from [a,b,c] -> [[a,0,0],[0,b,0],[0,0,c]]
            traceified = traceify(new_noise_magnitudes)
            generators = torch.cat([before_adding_new_terms, traceified], dim=0)
            return Zonotope(centers, generators)

        else:
            raise Exception

    __rmul__ = __mul__

    def get_stacked(self):
        return torch.cat([self.centers.unsqueeze(0), self.generators], dim=0)


# tested - works
# for affine layers no need to add new noise symbols
def AffineZonotope(Zono, layer, bias=None):
    # assert() #makes sure dimensions are compatible
    centers = Zono.centers
    generators = Zono.generators
    new_centers = centers @ layer

    if bias is not None:
        new_centers = new_centers + bias

    new_generators = generators @ layer

    return Zonotope(new_centers, new_generators)


def TanhZonotope(Zono):
    lb = Zono.get_lb()
    ub = Zono.get_ub()

    centers = Zono.centers
    generators = Zono.generators

    if torch.all(lb == ub):
        centers = torch.tanh(lb)
        generators = torch.zeros_like(centers).unsqueeze(0)
        return Zonotope(centers, generators)

    lambda_opt = torch.min(-torch.square(torch.tanh(lb)) + 1, -torch.square(torch.tanh(ub)) + 1)
    # print("lamba: ",lambda_opt)
    mu1 = 0.5 * (torch.tanh(ub) + torch.tanh(lb) - lambda_opt * (ub + lb))
    mu2 = 0.5 * (torch.tanh(ub) - torch.tanh(lb) - lambda_opt * (ub - lb))

    new_center = (lambda_opt * centers) + mu1
    new_generators_before = lambda_opt * generators

    traceified_mu2 = traceify(mu2)
    new_generators = torch.cat((new_generators_before, traceified_mu2), 0)

    return Zonotope(new_center, new_generators)


# Tested and works
def SoftPlus(x, beta=1):
    if type(x) in [int, float]:
        x = torch.tensor(x)
    return softplus(x, beta, threshold=100)


# Tested and works
def Sigmoid(x, beta=1):
    if type(x) in [int, float]:
        x = torch.tensor(x)
    return torch.sigmoid(x, beta)


# Tested and works
# inverse of a sigmoid function
def InverseSigmoid(x, beta=1):
    if type(x) in [int, float]:
        x = torch.tensor(x)
    return torch.div((torch.log(x + 1e-20) - torch.log(1 - x + 1e-20)), beta)


# Tested and works!
# Uses this form: Fast reliable interrogation of procedurally defined implicit surfaces using extended revised affine arithmetic
# https://www.researchgate.net/publication/220251211_Fast_reliable_interrogation_of_procedurally_defined_implicit_surfaces_using_extended_revised_affine_arithmetic
def SoftPlusZonoChebyshev(Zono, beta=1):
    # print("Softplus input zono number of noise symbols: ",Zono.get_num_noise_symbs())
    centers = Zono.centers
    generators = Zono.generators

    a = Zono.get_lb()  # these work
    b = Zono.get_ub()  # these work

    if torch.all(a == b):
        centers = SoftPlus(a, beta)
        generators = torch.zeros_like(centers).unsqueeze(0)  # This must have more than one dimension
        return Zonotope(centers, generators)

    fa = SoftPlus(a, beta)
    fb = SoftPlus(b, beta)
    alpha = (fb - fa) / (b - a)  # lambda_opt in DeepZ
    intercept = fa - (alpha * a)
    u = InverseSigmoid(alpha, beta)
    r = lambda x: alpha * x + intercept
    zeta = 0.5 * (SoftPlus(u, beta) + r(u)) - (alpha * u)  # mu1 in DeepZ
    delta = 0.5 * abs(SoftPlus(u, beta) - r(u))  # mu2 in DeepZ

    new_center = (alpha * centers) + zeta
    new_generators_before = alpha * generators

    traceified_delta = traceify(delta)
    new_generators = torch.cat((new_generators_before, traceified_delta), 0)
    res = Zonotope(new_center, new_generators)
    old_noise_symbols = Zono.get_num_noise_symbs()
    Zono.expand(res.get_num_noise_symbs() - old_noise_symbols)  # SIDE EFFECT
    return res


def ExpZonoChebyshev(Zono):
    # print("Softplus input zono number of noise symbols: ",Zono.get_num_noise_symbs())
    centers = Zono.centers
    generators = Zono.generators

    a = Zono.get_lb()  # these work
    b = Zono.get_ub()  # these work
    fa = torch.exp(a)
    fb = torch.exp(b)
    alpha = (fb - fa) / (b - a)  # lambda_opt in DeepZ
    intercept = fa - (alpha * a)
    u = torch.log(alpha)
    r = lambda x: alpha * x + intercept
    zeta = 0.5 * (torch.exp(u) + r(u)) - (alpha * u)  # mu1 in DeepZ
    delta = 0.5 * abs(torch.exp(u) - r(u))  # mu2 in DeepZ

    new_center = (alpha * centers) + zeta
    new_generators_before = alpha * generators

    traceified_delta = traceify(delta)
    new_generators = torch.cat((new_generators_before, traceified_delta), 0)
    res = Zonotope(new_center, new_generators)
    old_noise_symbols = Zono.get_num_noise_symbs()
    Zono.expand(res.get_num_noise_symbs() - old_noise_symbols)  # SIDE EFFECT!
    return res


def SigmoidZonotope(Zono):
    # print("Sigmoid input zono number of noise symbols: ",Zono.get_num_noise_symbs())
    lb = Zono.get_lb()  # these work
    ub = Zono.get_ub()  # these work

    centers = Zono.centers
    generators = Zono.generators

    if torch.all(lb == ub):
        centers = torch.sigmoid(lb)
        generators = torch.zeros_like(centers).unsqueeze(0)  # This must have more than one dimension
        return Zonotope(centers, generators)

    lambda_opt = torch.min(torch.sigmoid(lb) * (1 - torch.sigmoid(lb)), torch.sigmoid(ub) * (1 - torch.sigmoid(ub)))

    mu1 = 0.5 * (torch.sigmoid(ub) + torch.sigmoid(lb) - lambda_opt * (ub + lb))
    mu2 = 0.5 * (torch.sigmoid(ub) - torch.sigmoid(lb) - lambda_opt * (ub - lb))

    new_center = (lambda_opt * centers) + mu1
    new_generators_before = lambda_opt * generators

    traceified_mu2 = traceify(mu2)
    new_generators = torch.cat((new_generators_before, traceified_mu2), 0)

    # return Zonotope(new_center,new_generators)
    res = Zonotope(new_center, new_generators)
    old_noise_symbols = Zono.get_num_noise_symbs()
    Zono.expand(res.get_num_noise_symbs() - old_noise_symbols)  # SIDE EFFECT
    return res


def HyperDualIntervalToDualZonotope(hdi):
    real = IntervalsToZonotope(hdi.real_l.flatten(), hdi.real_u.flatten())
    dual = IntervalsToZonotope(hdi.e1_l.flatten(), hdi.e1_u.flatten())
    dual.generators = torch.cat([torch.zeros((real.generators.shape[0], real.generators.shape[1])), dual.generators])

    dual_num_noise_terms = dual.get_num_noise_symbs()
    real_num_noise_terms = real.get_num_noise_symbs()
    real.expand(dual_num_noise_terms - real_num_noise_terms)
    return DualZonotope(real.centers, real.generators, dual.centers, dual.generators)


class DualZonotope:
    def __init__(self, real_centers, real_coefs, dual_centers, dual_coefs):
        assert (real_coefs.shape == dual_coefs.shape)
        self.real = Zonotope(real_centers, real_coefs)
        self.dual = Zonotope(dual_centers, dual_coefs)

    def __str__(self):
        return ("Real: \n" + self.real.__str__() + "\nDual: \n" + self.dual.__str__() + "\n")

    def clone(self):
        new_real = self.real.clone()
        new_dual = self.dual.clone()
        return DualZonotope(new_real.centers, new_real.generators, new_dual.centers, new_dual.generators)

    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            other = other.flatten()
            return DualZonotope(self.real.centers + other, self.real.generators, self.dual.centers,
                                self.dual.generators)
        elif isinstance(other, self.__class__):
            r = self.real + other.real
            d = self.dual + other.dual

            return DualZonotope(r.centers, r.generators, d.centers, d.generators)
        else:
            raise Exception

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return DualZonotope(-self.real.centers, self.real.generators, -self.dual.centers, self.dual.generators)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DualZonotope(self.real.centers * other, self.real.generators * other, self.dual.centers * other,
                                self.dual.generators * other)
        elif isinstance(other, torch.Tensor):
            other = other.flatten()
            return DualZonotope(self.real.centers * other, self.real.generators * other, self.dual.centers * other,
                                self.dual.generators * other)
        else:
            raise Exception

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return DualZonotope(self.real.centers / other, self.real.generators / other, self.dual.centers / other,
                                self.dual.generators / other)
        elif isinstance(other, torch.Tensor):
            other = other.flatten()
            return DualZonotope(self.real.centers / other, self.real.generators / other, self.dual.centers / other,
                                self.dual.generators / other)
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{self.__class__}' and '{type(other)}'")


# assumes the layer and bias are torch tensors
def AffineDualZonotope(DZ, layer, bias=None):
    # assert() #asserts that the dimensions of the layer match the number of variables in the zonotope
    real = AffineZonotope(DZ.real, layer)

    if bias is not None:
        real = real + bias

    dual = AffineZonotope(DZ.dual, layer)
    return DualZonotope(real.centers, real.generators, dual.centers, dual.generators)


# Tested -works
def SmoothReluDualZonotope(DualZono):
    smoothrelu_real = SoftPlusZonoChebyshev(DualZono.real)  # HAS SIDE EFFECT
    smoothrelu_deriv = SigmoidZonotope(DualZono.real)  # HAS SIDE EFFECT

    dual = smoothrelu_deriv * DualZono.dual  # HAS SIDE EFFECT

    # go back and expand real part's number of noise symbols to match
    dual_num_noise_terms = dual.get_num_noise_symbs()
    real_num_noise_terms = smoothrelu_real.get_num_noise_symbs()
    smoothrelu_real.expand(dual_num_noise_terms - real_num_noise_terms)

    return DualZonotope(smoothrelu_real.centers, smoothrelu_real.generators, dual.centers, dual.generators)


def TanhDualZonotope(DualZono):
    tanh_real = TanhZonotope(DualZono.real)

    tanh_deriv = -(tanh_real * tanh_real) + 1
    # expand the dual part's number of noise symbols to match
    dual = tanh_deriv * DualZono.dual

    # go back and expand real part's number of noise symbols to match
    dual_num_noise_terms = dual.get_num_noise_symbs()
    real_num_noise_terms = tanh_real.get_num_noise_symbs()
    tanh_real.expand(dual_num_noise_terms - real_num_noise_terms)

    return DualZonotope(tanh_real.centers, tanh_real.generators, dual.centers, dual.generators)


def SigmoidDualZonotope(DualZono):
    sigmoid_real = SigmoidZonotope(DualZono.real)

    sigmoid_deriv = sigmoid_real * (-sigmoid_real + 1)
    # expand the dual part's number of noise symbols to match
    dual = sigmoid_deriv * DualZono.dual

    # go back and expand real part's number of noise symbols to match
    dual_num_noise_terms = dual.get_num_noise_symbs()
    real_num_noise_terms = sigmoid_real.get_num_noise_symbs()
    sigmoid_real.expand(dual_num_noise_terms - real_num_noise_terms)

    return DualZonotope(sigmoid_real.centers, sigmoid_real.generators, dual.centers, dual.generators)


# All this stuff is for maximizing the matrix norm of the Zonotope over-approx of the Jacobian
def format_str(st):
    l = list(st)
    return torch.tensor([float(-1) if x == "0" else float(x) for x in l])


def add_leading_one(mat):
    rows = mat.shape[0]
    new = torch.zeros((rows + 1, rows + 1))
    new[0, 0] = 1.
    new[1:, 1:] = mat
    return new


def get_identities(n):
    # generate all bitstrings
    all_bin_str = ["".join(seq) for seq in itertools.product("01", repeat=n)]
    # print(all_bin_str)
    formatted = [format_str(x) for x in all_bin_str]
    # print(formatted)
    traceified = [traceify(x) for x in formatted]
    final_tensor = torch.stack([add_leading_one(x) for x in traceified])
    return final_tensor


# checks if a certain row is always zero for all matrices in the list
def check_zero_rows(lst_of_mats):
    S = torch.sum(torch.sum(torch.abs(torch.stack(lst_of_mats)), dim=0), dim=1)
    res = (S != 0)
    return res


# https://discuss.pytorch.org/t/filter-out-undesired-rows/28933
def filter_zeros(a, ind):
    return a[ind, :]


def filter_lst(lst_of_mats):
    inds = check_zero_rows(lst_of_mats)
    return [filter_zeros(x, inds) for x in lst_of_mats]


def filter_dual_zono(dz):
    lst = [dz.real.generators, dz.dual.generators]
    inds = check_zero_rows(lst)
    real_generators = filter_zeros(dz.real.generators, inds)
    dual_generators = filter_zeros(dz.dual.generators, inds)
    return DualZonotope(dz.real.centers, real_generators, dz.dual.centers, dual_generators)


class ZonoJacobian:
    def __init__(self, lst_of_zonos):  # list has length = num of inputs of NN function
        self.lst_of_zonos = lst_of_zonos
        lst_as_tensors = filter_lst([x.get_stacked() for x in lst_of_zonos])
        # print("\n\n\n")
        # print(self.lst_of_zonos[0])
        # print(self.lst_of_zonos[1])
        # print("\n\n\n")
        self.stacked = torch.stack(lst_as_tensors)
        self.transpose_stacked = torch.stack(
            [x.t() for x in lst_as_tensors])  # torch.stack([x.get_stacked().t() for x in lst_of_zonos])
        self.num_noise = lst_as_tensors[0].shape[
                             0] - 1  # we have filtered out all zero noise symbols to reduce dimensionality
        self.num_outputs = self.lst_of_zonos[0].get_num_vars()

    def L_inf(self):
        dummy_added = self.transpose_stacked.unsqueeze(0)
        # print(dummy_added.shape)
        identities = get_identities(self.num_noise)
        # print(identities.shape)
        identities = identities.unsqueeze(1)
        # print(identities.shape)
        masked = dummy_added @ identities
        # print(masked.shape)
        sum_along_generators = torch.sum(masked, 3)
        # print(sum_along_generators.shape)
        sum_along_cols = torch.sum(sum_along_generators, 1)
        # print(sum_along_cols.shape)
        return torch.max(sum_along_cols)

    def Print(self):
        print("ZonoJacobian")
        for l in self.lst_of_zonos:
            print(l.get_lb())
            print(l.get_ub())
        print("\n")


# given a real input interval and a function, compute the zonotope jacobian over that region
def ComputeZonotopeJacobian(real_l, real_u, f):
    assert (real_l.shape == real_u.shape)
    numInputs = len(real_l)
    JacobianCols = []
    for i in range(numInputs):
        real = IntervalsToZonotope(real_l, real_u)
        d = torch.zeros(numInputs)
        d[i] = 1
        dual = IntervalsToZonotope(d, d)
        inputDZ = DualZonotope(real.centers, real.generators, dual.centers, dual.generators)
        outputDZ = f(inputDZ)
        JacobianCols.append(outputDZ.dual)
    Jacobian = ZonoJacobian(JacobianCols)
    return Jacobian


def split_zonotope(zono, k):
    num_vars = zono.centers.size()[0]
    assert (num_vars % k == 0)
    num_splits = int(num_vars / k)

    center_chunks = [zono.centers[(i * k):(i * k) + k] for i in range(num_splits)]
    generator_chunks = [zono.generators[:, (i * k):(i * k) + k] for i in range(num_splits)]
    zonos = [Zonotope(center_chunks[i], generator_chunks[i]) for i in range(num_splits)]
    return zonos


def Conv2dDualZonotope(x: DualZonotope, conv: torch.nn.Conv2d, shape: (int, int, int)):
    if not isinstance(conv, torch.nn.Conv2d):
        raise ValueError("Unsupported conv layer type!")

    input_size = shape[1:]

    output_size = [
        (input_size[i] + 2 * conv.padding[i] - conv.kernel_size[i]) // conv.stride[i]
        + 1
        for i in [0, 1]
    ]

    out_shape = (conv.out_channels, output_size[0], output_size[1])

    linear_ = torch_conv_layer_to_affine(conv, input_size)
    return AffineDualZonotope(x, linear_.weight.T) + linear_.bias, out_shape
