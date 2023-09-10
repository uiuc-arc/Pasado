import torch
from sklearn import linear_model
from SimpleZono import *


# lx, ly, ux, uy are TENSORS/VECTORS who have as many columns as neurons/variables in a layer
def get_linspace(lx, ux, ly, uy, STEPS=5):
    xs = [torch.linspace(lx[i].item(), ux[i].item(), steps=STEPS) for i in range(lx.shape[0])]
    ys = [torch.linspace(ly[i].item(), uy[i].item(), steps=STEPS) for i in range(ly.shape[0])]

    zs = [torch.cartesian_prod(xs[i], ys[i]) for i in range(len(xs))]
    return zs

# sigmoid(x)*(1-sigmoid(x))*y
def sigmoid_prime_times_y(xy_tensor):
    xs = xy_tensor[:, 0]
    ys = xy_tensor[:, 1]
    return (1.0 - torch.sigmoid(xs)) * (torch.sigmoid(xs) * ys)


def softplus_prime_times_y(xy_tensor):
    xs = xy_tensor[:, 0]
    ys = xy_tensor[:, 1]
    return (torch.sigmoid(xs) * ys)


# converts a list of tensors like [tensor(3.4), tensor(11.2), tensor(7)]
# to a tensor of the form tensor([3.4, 11.2, 7])
def lst_as_tensor(lst_of_tensors):
    res = torch.stack(lst_of_tensors)
    return res

def PlanarApproximation(A, B, C, xy_tensor):
    xs = xy_tensor[:, 0]
    ys = xy_tensor[:, 1]
    return (A * xs + B * ys + C)  # this does elementwise multiplication (thats what we want)


# works
def inv_sigmoid(y):
    return torch.log(y / (1. - y))


# useful for checking/debugging the inverse function
def sigmoid_double_prime(xs):
    return (1.0 - torch.sigmoid(xs)) * (torch.sigmoid(xs)) * (1 - 2 * torch.sigmoid(xs))


"""
#sklearn version so probably inefficient compared to pytorch  ltsq function
#https://pytorch.org/docs/stable/generated/torch.lstsq.html
def lin_reg(input_points,zs):
	#zs = f(input_points)
	ols = linear_model.LinearRegression()
	model = ols.fit(input_points, zs)
	A,B = model.coef_
	C = model.intercept_
	return (A,B,C)
"""


# torch vectorized version of linear regression
# https://medium.com/@rcorbish/linear-regression-using-pytorch-dcf0165e3a6e
def lin_reg_tensor(x, zs):
    # zs = f(x)
    xplusone = torch.cat((torch.ones(x.size(0), 1), x), 1)
    R = torch.linalg.lstsq(xplusone, zs, driver='gelsd').solution  # torch's least squares solver for linear regression
    R = R[0:xplusone.size(1)]

    #	R = torch.linalg.lstsq(xplusone,zs).solution
    #	shuffled = torch.ones(R.shape)
    #	shuffled[0:-1]=R[1:]
    #	shuffled[-1]=R[0] #THIS IS BECAUSE TORCH ORIGINALLY PUTS THE INTERCEPT FIRST
    #	return shuffled
    #	yh = xplusone.mm( R )
    #	print("R ",R)
    return R


# tensorized/vectorized version (torch is overloaded for the ** operator)
def cuberoot(x):
    return (x ** (0.33333))


# tensorized/vectorized version
def inverse_poly_tensor(y):
    Q = torch.tensor(-1 / 12.)
    R = 0.25 * y

    R = R.type(torch.complex64)

    J = torch.tensor(1.j)

    sqrt_arg = ((Q ** 3) + (R ** 2))

    sqrt_arg = sqrt_arg.type(torch.complex64)

    S = cuberoot(R + torch.sqrt(sqrt_arg))
    T = cuberoot(R - torch.sqrt(sqrt_arg))

    x1 = S + T + 0.5

    sqrt_3_by_2 = torch.sqrt(torch.tensor(3.)) / 2.
    x2 = -0.5 * (S + T) + 0.5 + (sqrt_3_by_2 * J) * (S - T)
    x3 = -0.5 * (S + T) + 0.5 - (sqrt_3_by_2 * J) * (S - T)

    return (x1, x2, x3)



# takes in a bool tensor and returns a float tensor where false gets mapped to NaN and true becomes 1.0
def bool_to_nan(x):
    b1 = torch.isreal(x)
    b1 = b1.float()
    b1[b1 < 0.5] = float('nan')
    return b1


def inf_to_nan(x):
    b = x
    b[torch.isinf(b)] = float('nan')
    return b


# takes a tensor x and returns a new tensor where any x_i not in the range [l,u] is replaced with NAN
def filter_range(l, u, x):
    b = x
    guard = torch.logical_not(torch.logical_and((b <= u), (b >= l)))
    b[guard] = float('nan')
    return b


def inverse_sigmoid_2nd_deriv(y):
    (x1, x2, x3) = inverse_poly_tensor(y)

    b1 = bool_to_nan(x1)  # sets any complex root to just NaN
    x1 = x1.real * b1

    b2 = bool_to_nan(x2)  # sets any complex root to just NaN
    x2 = x2.real * b2

    b3 = bool_to_nan(x3)  # sets any complex root to just NaN
    x3 = x3.real * b3

    inverted_x1 = (inv_sigmoid(x1))  # inv_sigmoid applied to a NaN is still just NaN
    inverted_x2 = (inv_sigmoid(x2))
    inverted_x3 = (inv_sigmoid(x3))

    return (inverted_x1, inverted_x2, inverted_x3)


def inverse_quadratic_tensor(y):
    inside_sqrt = 1.0 - (4.0 * y)
    inside_sqrt = inside_sqrt.type(torch.complex64)

    radical = torch.sqrt(inside_sqrt)

    x1 = radical * (-0.5) + 0.5
    x2 = (-radical * (-0.5)) + 0.5
    return (x1, x2)


def inverse_softplus_2nd_deriv(y):
    (x1, x2) = inverse_quadratic_tensor(y)

    b1 = bool_to_nan(x1)  # sets any complex root to just NaN
    x1 = x1.real * b1

    b2 = bool_to_nan(x2)  # sets any complex root to just NaN
    x2 = x2.real * b2

    inverted_x1 = (inv_sigmoid(x1))  # inv_sigmoid applied to a NaN is still just NaN
    inverted_x2 = (inv_sigmoid(x2))

    return (inverted_x1, inverted_x2)


# TESTED - seems correct
# evaluate this at the corners
def objective_fn(A, B, C, xy_tensor):
    return sigmoid_prime_times_y(xy_tensor) - PlanarApproximation(A, B, C, xy_tensor)


def objective_fn_softplus(A, B, C, xy_tensor):
    return softplus_prime_times_y(xy_tensor) - PlanarApproximation(A, B, C, xy_tensor)


# CHECKS BOTH F AND -F at the corners
# where F = (f'(x)y-(Ax+By+C))
# and  -F = -(f'(x)y-(Ax+By+C)) = (Ax+By+C)-(f'(x)y)
def check_corners(ABCs, lx, ux, ly, uy):
    maxs = []
    for i in range(lx.size()[0]):
        corner1 = torch.tensor([lx[i], ly[i]])

        corner2 = torch.tensor([lx[i], uy[i]])

        corner3 = torch.tensor([ux[i], ly[i]])

        corner4 = torch.tensor([ux[i], uy[i]])

        all_corners = torch.stack((corner1, corner2, corner3, corner4))
        # print("all corners ",all_corners)
        evaluation = objective_fn(ABCs[i][0], ABCs[i][1], ABCs[i][2], all_corners)
        # print("Evaluation of the corners ",evaluation)

        maxs.append(torch.max(torch.max(evaluation), -torch.min(evaluation)))

    return maxs


def check_corners_softplus(ABCs, lx, ux, ly, uy):
    maxs = []
    for i in range(lx.size()[0]):
        corner1 = torch.tensor([lx[i], ly[i]])

        corner2 = torch.tensor([lx[i], uy[i]])

        corner3 = torch.tensor([ux[i], ly[i]])

        corner4 = torch.tensor([ux[i], uy[i]])

        all_corners = torch.stack((corner1, corner2, corner3, corner4))
        # print("all corners ",all_corners)
        evaluation = objective_fn_softplus(ABCs[i][0], ABCs[i][1], ABCs[i][2], all_corners)
        # print("Evaluation of the corners ",evaluation)

        maxs.append(torch.max(torch.max(evaluation), -torch.min(evaluation)))

    return maxs


# solves the problem
# max xi in [l_xi,u_xi] sigmoid'(xi)yi - (Ai*xi + Bi*yi + Ci) where yi is fixed as either l_yi or u_yi
def check_nonlinear_boundary(lx, ux, ly, uy, ABCs):
    A_by_ly = torch.tensor([ABCs[i][0] / ly[i] for i in range(
        len(ABCs))])
    A_by_uy = torch.tensor([ABCs[i][0] / uy[i] for i in range(len(ABCs))])
    A_by_ly = (inf_to_nan(A_by_ly))  # Checks the case ly=0 (meaning A/ly is undefined)
    A_by_uy = (inf_to_nan(A_by_uy))

    l = -1.0 / (6.0 * np.sqrt(3.0))
    u = -l

    A_by_ly = filter_range(l, u, A_by_ly)  # checks the case A/ly is NOT in [-.1,.1] (not in image of sigmoid''(x))
    A_by_uy = filter_range(l, u, A_by_uy)

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1, root2, root3 = inverse_sigmoid_2nd_deriv(
        A_by_ly)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1 = filter_range(lx, ux, root1)
    root2 = filter_range(lx, ux, root2)
    root3 = filter_range(lx, ux, root3)

    # need to evaluate the original objective at these roots
    root1_paired = torch.stack((root1, ly), dim=1)
    root2_paired = torch.stack((root2, ly), dim=1)
    root3_paired = torch.stack((root3, ly), dim=1)

    roots_paired_stacked = torch.stack((root1_paired, root2_paired, root3_paired), dim=1)

    maxs = []
    for i in range(lx.size()[0]):

        evaluation = objective_fn(ABCs[i][0], ABCs[i][1], ABCs[i][2], roots_paired_stacked[i])
        evaluation = torch.nan_to_num(evaluation, -float("Inf"))
        maxs.append((torch.max(evaluation)))

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1_uy, root2_uy, root3_uy = inverse_sigmoid_2nd_deriv(
        A_by_uy)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1_uy = filter_range(lx, ux, root1_uy)
    root2_uy = filter_range(lx, ux, root2_uy)
    root3_uy = filter_range(lx, ux, root3_uy)

    # need to evaluate the original objective at these roots
    root1_uy_paired = torch.stack((root1_uy, uy), dim=1)
    root2_uy_paired = torch.stack((root2_uy, uy), dim=1)
    root3_uy_paired = torch.stack((root3_uy, uy), dim=1)

    roots_uy_paired_stacked = torch.stack((root1_uy_paired, root2_uy_paired, root3_uy_paired), dim=1)

    maxs_uy = []
    for i in range(lx.size()[0]):

        evaluation_uy = objective_fn(ABCs[i][0], ABCs[i][1], ABCs[i][2], roots_uy_paired_stacked[i])
        evaluation_uy = torch.nan_to_num(evaluation_uy, -float("Inf"))
        maxs_uy.append((torch.max(evaluation_uy)))

    return maxs, maxs_uy


def check_nonlinear_boundary_softplus(lx, ux, ly, uy, ABCs):
    A_by_ly = torch.tensor([ABCs[i][0] / ly[i] for i in range(
        len(ABCs))])
    A_by_uy = torch.tensor([ABCs[i][0] / uy[i] for i in range(len(ABCs))])
    A_by_ly = (inf_to_nan(A_by_ly))  # Checks the case ly=0 (meaning A/ly is undefined)
    A_by_uy = (inf_to_nan(A_by_uy))

    l = 0
    u = 0.25

    A_by_ly = filter_range(l, u, A_by_ly)  # checks the case A/ly is NOT in [-.1,.1] (not in image of sigmoid''(x))
    A_by_uy = filter_range(l, u, A_by_uy)

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1, root2 = inverse_softplus_2nd_deriv(
        A_by_ly)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1 = filter_range(lx, ux, root1)
    root2 = filter_range(lx, ux, root2)

    # need to evaluate the original objective at these roots
    root1_paired = torch.stack((root1, ly), dim=1)
    root2_paired = torch.stack((root2, ly), dim=1)

    roots_paired_stacked = torch.stack((root1_paired, root2_paired), dim=1)

    maxs = []
    for i in range(lx.size()[0]):

        evaluation = objective_fn_softplus(ABCs[i][0], ABCs[i][1], ABCs[i][2], roots_paired_stacked[i])
        evaluation = torch.nan_to_num(evaluation, -float("Inf"))
        maxs.append((torch.max(evaluation)))

    # NEED TO DO ALL THIS AGAIN FOR A_by_uy
    root1_uy, root2_uy = inverse_softplus_2nd_deriv(
        A_by_uy)  # because of the nature of sigmoid''(x), only at most 2 of these will be real

    # checks if the roots (where the extrema is reached) is within [lx,ux], if not the root is replaced with NaN
    root1_uy = filter_range(lx, ux, root1_uy)
    root2_uy = filter_range(lx, ux, root2_uy)

    # need to evaluate the original objective at these roots
    root1_uy_paired = torch.stack((root1_uy, uy), dim=1)
    root2_uy_paired = torch.stack((root2_uy, uy), dim=1)

    roots_uy_paired_stacked = torch.stack((root1_uy_paired, root2_uy_paired), dim=1)

    maxs_uy = []
    for i in range(lx.size()[0]):

        evaluation_uy = objective_fn_softplus(ABCs[i][0], ABCs[i][1], ABCs[i][2], roots_uy_paired_stacked[i])
        evaluation_uy = torch.nan_to_num(evaluation_uy, -float("Inf"))
        maxs_uy.append((torch.max(evaluation_uy)))

    return maxs, maxs_uy


"""
def check_corners_negated(ABCs,lx,ux,ly,uy):

	maxs = []
	for i in range(lx.size()[0]):
		corner1 = torch.tensor([lx[i],ly[i]])

		corner2 = torch.tensor([lx[i],uy[i]])

		corner3 = torch.tensor([ux[i],ly[i]])

		corner4 = torch.tensor([ux[i],uy[i]])

		all_corners = torch.stack((corner1,corner2,corner3,corner4))

		evaluation = (-objective_fn(ABCs[i][0],ABCs[i][1],ABCs[i][2],all_corners))

		maxs.append(torch.max(evaluation))


	return maxs
"""


def compute_max_error(lx, ux, ly, uy, ABCs):
    # checks along the linear function boundaries (by just checking the corners)
    errors = check_corners(ABCs, lx, ux, ly, uy)  # This seems to be correct

    # still need to check along the nonlinear boudnary, e.g.
    # f'(x)*l_y...
    # f'(x)*u_y...
    errors2_ly, errors2_uy = check_nonlinear_boundary(lx, ux, ly, uy, ABCs)
    max_errors = []
    assert (len(errors) == len(errors2_ly) and len(errors) == len(errors2_uy))
    for i in range(len(errors)):
        max_errors.append(torch.max(torch.max(errors[i], errors2_ly[i]), errors2_uy[i]))

    return max_errors


def compute_max_error_softplus(lx, ux, ly, uy, ABCs):
    # checks along the linear function boundaries (by just checking the corners)
    errors = check_corners_softplus(ABCs, lx, ux, ly, uy)  # This seems to be correct

    # still need to check along the nonlinear boudnary, e.g.
    # f'(x)*l_y...
    # f'(x)*u_y...
    errors2_ly, errors2_uy = check_nonlinear_boundary_softplus(lx, ux, ly, uy, ABCs)
    max_errors = []
    assert (len(errors) == len(errors2_ly) and len(errors) == len(errors2_uy))
    for i in range(len(errors)):
        max_errors.append(torch.max(torch.max(errors[i], errors2_ly[i]), errors2_uy[i]))

    return max_errors


def sigmoid_prime_product_tensor(x, y):
    lx = x.get_lb()
    ux = x.get_ub()
    ly = y.get_lb()
    uy = y.get_ub()

    xys = get_linspace(lx, ux, ly, uy)  # this is a list of tensors

    zs = [sigmoid_prime_times_y(xy) for xy in xys]

    lineqs = [lin_reg_tensor(xys[i], zs[i]) for i in
              range(len(zs))]  # list of tensors([A,B,C]) where Ax+By+C is the planar regression

    # this has to be 1,2,0 since the linreg function gives the constant offset (c) as the 0th element of the tuple
    ABCs = [(lineqs[i][1], lineqs[i][2], lineqs[i][0]) for i in range(len(lineqs))]
    ABCs = [(x[0] + np.random.normal(0, .0001), x[1], x[2]) for x in ABCs]
    assert (all([(x[0] > 0 or x[0] < 0) for x in ABCs]))
    As = torch.tensor([x[0] for x in ABCs])
    Bs = torch.tensor([x[1] for x in ABCs])
    Cs = torch.tensor([x[2] for x in ABCs])

    # Everything up to this point works (the linear regression and the lower and upper bounds of the input Zonotopes)
    errors = compute_max_error(lx, ux, ly, uy, ABCs)
    errors = lst_as_tensor(errors)
    errors = traceify(errors)

    x_centers = x.centers
    x_generators = x.generators

    y_centers = y.centers
    y_generators = y.generators

    new_centers = As * x_centers + Bs * y_centers + Cs
    new_generators = As * x_generators + Bs * y_generators

    # add in the fresh error symbols obtained by solving the optimization problem
    new_generators = torch.cat((new_generators, errors), 0)

    return Zonotope(new_centers, new_generators)


def softplus_prime_product_tensor(x, y):
    lx = x.get_lb()
    ux = x.get_ub()
    ly = y.get_lb()
    uy = y.get_ub()

    xys = get_linspace(lx, ux, ly, uy)  # this is a list of tensors

    zs = [softplus_prime_times_y(xy) for xy in xys]

    lineqs = [lin_reg_tensor(xys[i], zs[i]) for i in
              range(len(zs))]  # list of tensors([A,B,C]) where Ax+By+C is the planar regression

    # this has to be 1,2,0 since the linreg function gives the constant offset (c) as the 0th element of the tuple
    ABCs = [(lineqs[i][1], lineqs[i][2], lineqs[i][0]) for i in range(len(lineqs))]
    assert (all([(x[0] > 0 or x[0] < 0) for x in ABCs]))
    As = torch.tensor([x[0] for x in ABCs])
    Bs = torch.tensor([x[1] for x in ABCs])
    Cs = torch.tensor([x[2] for x in ABCs])

    # Everything up to this point works (the linear regression and the lower and upper bounds of the input Zonotopes)
    errors = compute_max_error_softplus(lx, ux, ly, uy, ABCs)
    errors = lst_as_tensor(errors)
    errors = traceify(errors)

    x_centers = x.centers
    x_generators = x.generators

    y_centers = y.centers
    y_generators = y.generators

    new_centers = As * x_centers + Bs * y_centers + Cs
    new_generators = As * x_generators + Bs * y_generators

    # add in the fresh error symbols obtained by solving the optimization problem
    new_generators = torch.cat((new_generators, errors), 0)

    return Zonotope(new_centers, new_generators)


# this is where the above transformer is actually integrated into the entire expression for the Dual Zonotope
def PreciseSigmoidDualZonotope(DualZono):
    old_num_syms = DualZono.real.get_num_noise_symbs()
    old_num_syms_dual = DualZono.dual.get_num_noise_symbs()
    assert (old_num_syms == old_num_syms_dual)

    sigmoid_real = SigmoidZonotope(DualZono.real)

    real_num_noise_terms = sigmoid_real.get_num_noise_symbs()

    # DualZono.real.expand(real_num_noise_terms-old_num_syms) #dont need this the SigmoidZonotope already executes this exact command inside that function
    if 0 < real_num_noise_terms - old_num_syms_dual:
        DualZono.dual.expand(real_num_noise_terms - old_num_syms_dual)  # DO need this

    dual = sigmoid_prime_product_tensor(DualZono.real, DualZono.dual)  # This has NO side effects on the other zonotopes

    dual_num_noise_terms = dual.get_num_noise_symbs()
    sigmoid_real.expand(dual_num_noise_terms - real_num_noise_terms)


    return DualZonotope(sigmoid_real.centers, sigmoid_real.generators, dual.centers, dual.generators)


# this is where the above transformer is actually integrated into the entire expression for the Dual Zonotope
def PreciseSoftplusDualZonotope(DualZono):
    old_num_syms = DualZono.real.get_num_noise_symbs()
    old_num_syms_dual = DualZono.dual.get_num_noise_symbs()
    assert (old_num_syms == old_num_syms_dual)

    # sigmoid_real = SigmoidZonotope(DualZono.real)
    smoothrelu_real = SoftPlusZonoChebyshev(DualZono.real)

    real_num_noise_terms = smoothrelu_real.get_num_noise_symbs()

    # DualZono.real.expand(real_num_noise_terms-old_num_syms) #dont need this the SigmoidZonotope already executes this exact command inside that function
    DualZono.dual.expand(real_num_noise_terms - old_num_syms_dual)  # DO need this

    dual = softplus_prime_product_tensor(DualZono.real,
                                         DualZono.dual)  # This has NO side effects on the other zonotopes

    dual_num_noise_terms = dual.get_num_noise_symbs()
    smoothrelu_real.expand(dual_num_noise_terms - real_num_noise_terms)


    return DualZonotope(smoothrelu_real.centers, smoothrelu_real.generators, dual.centers, dual.generators)
