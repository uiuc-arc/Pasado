import mpmath
import numpy as np
# from mpmath import mp, fdiv, fadd, fsub, fsum, fneg, fmul, fabs, sqrt, exp, log, sin
import torch
from affapy.aa import Affine
from sklearn.linear_model import LinearRegression
from CubicEquationSolver import solve
import math
from MixedAAIA import *


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def AddNewNoiseSymbol(self, d):
    return self._affineConstructor(1., 0., d)


def quotient_rule(x1, y1, x2, y2):
    return (((x2 * y1) - (x1 * y2)) / (x2 * x2))


def product_rule(x1, y1, x2, y2):
    return ((x1 * y2) + (x2 * y1))


def get_corners(lx1, ux1, ly1, uy1, lx2, ux2, ly2, uy2):
    x1s = np.linspace(lx1, ux1, 2)
    y1s = np.linspace(ly1, uy1, 2)
    x2s = np.linspace(lx2, ux2, 2)
    y2s = np.linspace(ly2, uy2, 2)
    xys = np.asarray(cartesian_product(x1s, y1s, x2s, y2s))
    return xys


#


def SynthesizedQuotientRuleTransformer(x1, y1, x2, y2, N=10):
    # get grid points
    lx1, ux1 = float(x1.interval.inf), float(x1.interval.sup)
    x1s = np.linspace(lx1, ux1, N)

    ly1, uy1 = float(y1.interval.inf), float(y1.interval.sup)
    y1s = np.linspace(ly1, uy1, N)

    lx2, ux2 = float(x2.interval.inf), float(x2.interval.sup)
    x2s = np.linspace(lx2, ux2, N)

    assert ((0 > ux2) or (lx2 > 0))  # needed to ensure that 0 isn't included in the range that we divide by

    ly2, uy2 = float(y2.interval.inf), float(y2.interval.sup)
    y2s = np.linspace(ly2, uy2, N)

    xys = np.asarray(cartesian_product(x1s, y1s, x2s, y2s))

    zs = [quotient_rule(a[0], a[1], a[2], a[3]) for a in xys]
    zs = np.asarray(zs)

    reg = LinearRegression().fit(xys, zs)

    A1, B1, A2, B2 = reg.coef_
    C1 = reg.intercept_

    sig = 0.00001
    A1 = A1 + np.random.normal(0, sig)
    A2 = A2 + np.random.normal(0, sig)
    B1 = B1 + np.random.normal(0, sig)
    B2 = B2 + np.random.normal(0, sig)

    # now just need new noise symbol  obtained by evaluating at corner points AND solving the cubic polynomial
    corners = get_corners(lx1, ux1, ly1, uy1, lx2, ux2, ly2, uy2)

    ground_truth = np.asarray([quotient_rule(a[0], a[1], a[2], a[3]) for a in corners])
    evaluation = np.asarray([((A1 * a[0]) + (B1 * a[1]) + (A2 * a[2]) + (B2 * a[3]) + C1) for a in corners])

    max_diff = max(abs(ground_truth - evaluation))
    # print(max_diff)
    root_pts = []
    for X1 in [lx1, ux1]:
        for Y1 in [ly1, uy1]:
            for Y2 in [ly2, uy2]:
                roots = solve(A2, 0, Y1, -2 * X1 * Y2)  # solve(A1,0,-Y1,2*X1*Y2)
                real_roots = [q for q in roots if np.isreal(q)]
                real_roots_constrained = [q2 for q2 in real_roots if ((q2 <= ux2) and (q2 >= lx2))]
                #				print(real_roots)
                # print(real_roots_constrained)
                if len(real_roots_constrained) > 0:
                    root_pts = root_pts + [(X1, Y1, x2_, Y2) for x2_ in real_roots_constrained]

    # evaluate the objective at the roots (which are critical points)
    if len(root_pts) > 0:
        for r in root_pts:
            diff = (abs(quotient_rule(r[0], r[1], r[2], r[3]) - (
                        (A1 * r[0]) + (B1 * r[1]) + (A2 * r[2]) + (B2 * r[3]) + C1)))
            if diff > max_diff:
                max_diff = diff

    res = (A1 * x1) + (B1 * y1) + (A2 * x2) + (B2 * y2) + C1
    res = AddNewNoiseSymbol(res, max_diff)
    return res


"""
#synthesizes a transformer for x/y
def SynthesizedQuotientTransformer(x,y,N=10):
	#get grid points
	lx, ux = float(x.interval.inf), float(x.interval.sup)
	xs = np.linspace(lx,ux,N)

	ly, uy = float(y.interval.inf), float(y.interval.sup)
	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))

	zs = [a[0]/a[1] for a in xys]

	reg = LinearRegression().fit(xys, zs)

	A1,B1 = reg.coef_
	C1 = reg.intercept_


	corners = [(lx,ly),(lx,uy),(ux,ly),(ux,uy)]

"""


def SynthesizedProductRuleTransformer(x1, y1, x2, y2, N=10):
    # get grid points
    lx1, ux1 = float(x1.interval.inf), float(x1.interval.sup)
    x1s = np.linspace(lx1, ux1, N)

    ly1, uy1 = float(y1.interval.inf), float(y1.interval.sup)
    y1s = np.linspace(ly1, uy1, N)

    lx2, ux2 = float(x2.interval.inf), float(x2.interval.sup)
    x2s = np.linspace(lx2, ux2, N)

    ly2, uy2 = float(y2.interval.inf), float(y2.interval.sup)
    y2s = np.linspace(ly2, uy2, N)

    xys = np.asarray(cartesian_product(x1s, y1s, x2s, y2s))

    zs = [product_rule(a[0], a[1], a[2], a[3]) for a in xys]
    zs = np.asarray(zs)

    reg = LinearRegression().fit(xys, zs)

    A1, B1, A2, B2 = reg.coef_
    C1 = reg.intercept_

    # now just need new noise symbol  obtained by evaluating at corner points AND solving the cubic polynomial
    corners = get_corners(lx1, ux1, ly1, uy1, lx2, ux2, ly2, uy2)

    ground_truth = np.asarray([product_rule(a[0], a[1], a[2], a[3]) for a in corners])
    evaluation = np.asarray([((A1 * a[0]) + (B1 * a[1]) + (A2 * a[2]) + (B2 * a[3]) + C1) for a in corners])

    max_diff = max(abs(ground_truth - evaluation))

    res = (A1 * x1) + (B1 * y1) + (A2 * x2) + (B2 * y2) + C1
    res = AddNewNoiseSymbol(res, max_diff)
    return res


def return_interval_quotient_rule(x1, y1, x2, y2):
    return ((y1.bounds * x2.bounds) - (x1.bounds * y2.bounds)) / (x2.bounds ** 2)


def SynthesizedQuotientRuleTransformer_mixed(x1, y1, x2, y2, N=10):
    # get grid points
    lx1, ux1 = float(x1.bounds.inf), float(x1.bounds.sup)
    x1s = np.linspace(lx1, ux1, N)

    ly1, uy1 = float(y1.bounds.inf), float(y1.bounds.sup)
    y1s = np.linspace(ly1, uy1, N)

    lx2, ux2 = float(x2.bounds.inf), float(x2.bounds.sup)
    x2s = np.linspace(lx2, ux2, N)

    assert ((0 > ux2) or (lx2 > 0))  # needed to ensure that 0 isn't included in the range that we divide by

    ly2, uy2 = float(y2.bounds.inf), float(y2.bounds.sup)
    y2s = np.linspace(ly2, uy2, N)

    xys = np.asarray(cartesian_product(x1s, y1s, x2s, y2s))

    zs = [quotient_rule(a[0], a[1], a[2], a[3]) for a in xys]
    zs = np.asarray(zs)

    reg = LinearRegression().fit(xys, zs)

    A1, B1, A2, B2 = reg.coef_
    C1 = reg.intercept_

    sig = 0.00001
    A1 = A1 + np.random.normal(0, sig)
    A2 = A2 + np.random.normal(0, sig)
    B1 = B1 + np.random.normal(0, sig)
    B2 = B2 + np.random.normal(0, sig)

    # now just need new noise symbol  obtained by evaluating at corner points AND solving the cubic polynomial
    corners = get_corners(lx1, ux1, ly1, uy1, lx2, ux2, ly2, uy2)

    ground_truth = np.asarray([quotient_rule(a[0], a[1], a[2], a[3]) for a in corners])
    evaluation = np.asarray([((A1 * a[0]) + (B1 * a[1]) + (A2 * a[2]) + (B2 * a[3]) + C1) for a in corners])

    max_diff = max(abs(ground_truth - evaluation))
    # print(max_diff)
    root_pts = []
    for X1 in [lx1, ux1]:
        for Y1 in [ly1, uy1]:
            for Y2 in [ly2, uy2]:
                roots = solve(A2, 0, Y1, -2 * X1 * Y2)  # solve(A1,0,-Y1,2*X1*Y2)
                real_roots = [q for q in roots if np.isreal(q)]
                real_roots_constrained = [q2 for q2 in real_roots if ((q2 <= ux2) and (q2 >= lx2))]
                #				print(real_roots)
                # print(real_roots_constrained)
                if len(real_roots_constrained) > 0:
                    root_pts = root_pts + [(X1, Y1, x2_, Y2) for x2_ in real_roots_constrained]

    # evaluate the objective at the roots (which are critical points)
    if len(root_pts) > 0:
        for r in root_pts:
            diff = (abs(quotient_rule(r[0], r[1], r[2], r[3]) - (
                        (A1 * r[0]) + (B1 * r[1]) + (A2 * r[2]) + (B2 * r[3]) + C1)))
            if diff > max_diff:
                max_diff = diff

    res = (A1 * x1.affine_form) + (B1 * y1.affine_form) + (A2 * x2.affine_form) + (B2 * y2.affine_form) + C1
    res = AddNewNoiseSymbol(res, max_diff)

    corner_evals = [quotient_rule(a[0], a[1], a[2], a[3]) for a in corners]

    # Need to check for 0 ub cases where 0 in [l_a,u_a] and l_b = 0
    # also need to check c=(l_a*l_d)/(l_b) for each corner (can rule out if this root is not in [c_l,c_u])
    bool_check = lambda a: ((a[1] != 0) and ((a[0] * a[3]) != 0) and (((2 * a[0] * a[3]) / (a[1])) >= lx2) and (
                ((2 * a[0] * a[3]) / (a[1])) <= ux2))
    root_evals = [quotient_rule(a[0], a[1], ((2 * a[0] * a[3]) / (a[1])), a[3]) for a in corners if bool_check(a)]
    corner_evals = corner_evals + root_evals

    lower = min(corner_evals)
    upper = max(corner_evals)
    refined_interval = Interval(lower, upper)
    #	refined_interval2 = return_interval_quotient_rule(x1,y1,x2,y2)
    #	print("\nours: ")
    #	print(refined_interval)
    #	print("vanilla itnerval arithmetic ")
    #	print(refined_interval2)

    #	refined_interval = interval_intersection(refined_interval,refined_interval2)
    return MixedAffine(res, refined_interval)


# def SynthesizedQuotientRuleTransformer_mixed(x1,y1,x2,y2,N=10):
#
# 	#get grid points
# 	lx1, ux1 = float(x1.bounds.inf), float(x1.bounds.sup)
# 	x1s = np.linspace(lx1,ux1,N)
#
# 	ly1, uy1 = float(y1.bounds.inf), float(y1.bounds.sup)
# 	y1s = np.linspace(ly1,uy1,N)
#
#
# 	lx2, ux2 = float(x2.bounds.inf), float(x2.bounds.sup)
# 	x2s = np.linspace(lx2,ux2,N)
#
# 	assert((0>ux2) or (lx2>0)) #needed to ensure that 0 isn't included in the range that we divide by
#
#
# 	ly2, uy2 = float(y2.bounds.inf), float(y2.bounds.sup)
# 	y2s = np.linspace(ly2,uy2,N)
#
# 	xys = np.asarray(cartesian_product(x1s,y1s,x2s,y2s))
#
# 	zs = [quotient_rule(a[0],a[1],a[2],a[3]) for a in xys]
# 	zs = np.asarray(zs)
#
# 	reg = LinearRegression().fit(xys, zs)
#
# 	A1,B1,A2,B2 = reg.coef_
# 	C1 = reg.intercept_
#
# 	sig = 0.00001
# 	A1 = A1 +  np.random.normal(0, sig)
# 	A2 = A2 +  np.random.normal(0, sig)
# 	B1 = B1 +  np.random.normal(0, sig)
# 	B2 = B2 +  np.random.normal(0, sig)
#
# 	#now just need new noise symbol  obtained by evaluating at corner points AND solving the cubic polynomial
# 	corners = get_corners(lx1,ux1,ly1,uy1,lx2,ux2,ly2,uy2)
#
# 	ground_truth = np.asarray([quotient_rule(a[0],a[1],a[2],a[3]) for a in corners])
# 	evaluation = np.asarray([((A1*a[0])+(B1*a[1])+(A2*a[2])+(B2*a[3])+C1) for a in corners])
#
# 	max_diff_above = max((ground_truth-evaluation))
# 	max_diff_below = max((-ground_truth+evaluation))
# 	#print(max_diff)
# 	root_pts = []
# 	for X1 in [lx1,ux1]:
# 		for Y1 in [ly1,uy1]:
# 			for Y2 in [ly2,uy2]:
# 				roots = solve(A2,0,Y1,-2*X1*Y2)
# 				real_roots = [q for q in roots if np.isreal(q)]
# 				real_roots_constrained = [q2 for q2 in real_roots if ((q2<= ux2) and (q2>=lx2))]
# #				print(real_roots)
# 				#print(real_roots_constrained)
# 				if len(real_roots_constrained) > 0:
# 					root_pts = root_pts + [(X1,Y1,x2_,Y2) for x2_ in real_roots_constrained]
#
# 	#evaluate the objective at the roots (which are critical points)
# 	if len(root_pts) > 0:
# 		for r in root_pts:
# 			diff_above = ((quotient_rule(r[0],r[1],r[2],r[3])-((A1*r[0])+(B1*r[1])+(A2*r[2])+(B2*r[3])+C1)))
# 			diff_below = ((-quotient_rule(r[0],r[1],r[2],r[3])+((A1*r[0])+(B1*r[1])+(A2*r[2])+(B2*r[3])+C1)))
# 			if diff_above > max_diff_above:
# 				max_diff_above = diff_above
# 			if diff_below > max_diff_below:
# 				max_diff_below = diff_below
#
#
# 	assert(max_diff_above>0)
# 	assert(max_diff_below>0)
# 	max_val = ((max_diff_above+max_diff_below)*0.5)
# 	C1 = (C1 - max_diff_below) + max_val
# 	res = (A1*x1.affine_form)+(B1*y1.affine_form)+(A2*x2.affine_form)+(B2*y2.affine_form)+C1
# 	res = AddNewNoiseSymbol(res,max_val)
#
# 	corner_evals = [quotient_rule(a[0],a[1],a[2],a[3]) for a in corners]
#
# 	#Need to check for 0 ub cases where 0 in [l_a,u_a] and l_b = 0
# 	#also need to check c=(l_a*l_d)/(l_b) for each corner (can rule out if this root is not in [c_l,c_u])
# 	bool_check = lambda a : ((a[1] != 0) and ((a[0]*a[3]) != 0) and  (((2*a[0]*a[3])/(a[1])) >= lx2) and  (((2*a[0]*a[3])/(a[1])) <= ux2))
# 	root_evals = [quotient_rule(a[0],a[1],((2*a[0]*a[3])/(a[1])),a[3]) for a in corners if  bool_check(a)]
# 	corner_evals = corner_evals + root_evals
#
# 	lower = min(corner_evals)
# 	upper = max(corner_evals)
#
# 	#case 3.2 for intervals
# 	if (((lx1<=0) and (ux1>=0)) and ((ly2<=0) and (uy2>=0))) and (ly1==0 or uy1==0):
# 		lower = min(lower,0)
# 		upper = max(upper,0)
#
# 	refined_interval = Interval(lower,upper)
# #	refined_interval2 = return_interval_quotient_rule(x1,y1,x2,y2)
# #	print("\nours: ")
# #	print(refined_interval)
# #	print("vanilla itnerval arithmetic ")
# #	print(refined_interval2)
#
#
# #	refined_interval = interval_intersection(refined_interval,refined_interval2)
# 	return MixedAffine(res,refined_interval)
#
#
#
#
def SynthesizedProductRuleTransformer_mixed(x1, y1, x2, y2, N=10):
    # get grid points
    lx1, ux1 = float(x1.bounds.inf), float(x1.bounds.sup)
    x1s = np.linspace(lx1, ux1, N)

    ly1, uy1 = float(y1.bounds.inf), float(y1.bounds.sup)
    y1s = np.linspace(ly1, uy1, N)

    lx2, ux2 = float(x2.bounds.inf), float(x2.bounds.sup)
    x2s = np.linspace(lx2, ux2, N)

    ly2, uy2 = float(y2.bounds.inf), float(y2.bounds.sup)
    y2s = np.linspace(ly2, uy2, N)

    xys = np.asarray(cartesian_product(x1s, y1s, x2s, y2s))

    zs = [product_rule(a[0], a[1], a[2], a[3]) for a in xys]
    zs = np.asarray(zs)

    reg = LinearRegression().fit(xys, zs)

    A1, B1, A2, B2 = reg.coef_
    C1 = reg.intercept_

    # now just need new noise symbol  obtained by evaluating at corner points AND solving the cubic polynomial
    corners = get_corners(lx1, ux1, ly1, uy1, lx2, ux2, ly2, uy2)

    ground_truth = np.asarray([product_rule(a[0], a[1], a[2], a[3]) for a in corners])
    evaluation = np.asarray([((A1 * a[0]) + (B1 * a[1]) + (A2 * a[2]) + (B2 * a[3]) + C1) for a in corners])

    max_diff = max(abs(ground_truth - evaluation))

    res = (A1 * x1.affine_form) + (B1 * y1.affine_form) + (A2 * x2.affine_form) + (B2 * y2.affine_form) + C1
    res = AddNewNoiseSymbol(res, max_diff)

    # interval refinement:
    corner_evals = [product_rule(a[0], a[1], a[2], a[3]) for a in corners]
    lower = min(corner_evals)
    upper = max(corner_evals)
    refined_interval = Interval(lower, upper)

    return MixedAffine(res, refined_interval)
