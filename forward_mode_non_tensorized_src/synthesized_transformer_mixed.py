import mpmath
import numpy as np
#from mpmath import mp, fdiv, fadd, fsub, fsum, fneg, fmul, fabs, sqrt, exp, log, sin
import torch
from affapy.aa import Affine
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import math
import invert_gaussian_deriv as ig
import CubicEquationSolver
from MixedAAIA import *
from synthesized_transformer import *



def TanhDeepZ_mixed(self):
	lx, ux = float(self.bounds.inf), float(self.bounds.sup)

	if lx==ux:
		new_affine_form = Affine([np.tanh(lx),np.tanh(lx)]) 
		new_bounds = Interval(np.tanh(lx),np.tanh(lx))
		return MixedAffine(new_affine_form,new_bounds)

	lambda_opt = min(tanh_deriv(lx),tanh_deriv(ux))

	mu1 = 0.5*(np.tanh(ux)+np.tanh(lx)-(lambda_opt*(ux+lx)))

	mu2 = 0.5*(np.tanh(ux)-np.tanh(lx)-(lambda_opt*(ux-lx)))

	alpha = mpmath.mpmathify(lambda_opt)
	dzeta = mpmath.mpmathify(mu1)
	delta = mpmath.mpmathify(mu2)

	new_affine_form = self.affine_form._affineConstructor(alpha, dzeta, delta)
	new_bounds = Interval(np.tanh(lx),np.tanh(ux))
	return MixedAffine(new_affine_form,new_bounds)


def SigmoidDeepZ_mixed(self):
	lx, ux = float(self.bounds.inf), float(self.bounds.sup) 

	if lx==ux:
		new_affine_form = Affine([sigmoid(lx),sigmoid(lx)])
		new_bounds = Interval(sigmoid(lx),sigmoid(lx))
		return MixedAffine(new_affine_form,new_bounds)

	lambda_opt = min(sigmoid_deriv(lx),sigmoid_deriv(ux))

	mu1 = 0.5*(sigmoid(ux)+sigmoid(lx)-(lambda_opt*(ux+lx)))

	mu2 = 0.5*(sigmoid(ux)-sigmoid(lx)-(lambda_opt*(ux-lx)))

	alpha = mpmath.mpmathify(lambda_opt)
	dzeta = mpmath.mpmathify(mu1)
	delta = mpmath.mpmathify(mu2)

	new_affine_form = self.affine_form._affineConstructor(alpha, dzeta, delta)
	new_bounds = Interval(sigmoid(lx),sigmoid(ux))
	return MixedAffine(new_affine_form,new_bounds)



def ExpChebyshev_mixed(self):
	lx, ux = float(self.bounds.inf), float(self.bounds.sup)

	if lx==ux:
		new_affine_form = Affine([np.exp(lx),np.exp(lx)]) 
		new_bounds = Interval(np.exp(lx),np.exp(lx))
		return MixedAffine(new_affine_form,new_bounds)

	a = lx
	b = ux

	fa = np.exp(a)
	fb = np.exp(b)

	alpha = (fb-fa)/(b-a)

	u = np.log(alpha)

	r = lambda x : alpha*x+((-alpha*b)+(fb))

	dzeta = ((np.exp(u)+r(u))/2.0)-(alpha*u)

	delta = abs(np.exp(u)-r(u))/2.0

	alpha = mpmath.mpmathify(alpha)
	dzeta = mpmath.mpmathify(dzeta)
	delta = mpmath.mpmathify(delta)


	new_affine_form = self.affine_form._affineConstructor(alpha, dzeta, delta)
	new_bounds = Interval(np.exp(lx),np.exp(ux))
	return MixedAffine(new_affine_form,new_bounds)




def SqrtChebyshev_mixed(self):
	lx, ux = float(self.bounds.inf), float(self.bounds.sup)

	if lx==ux:
		new_affine_form = Affine([np.sqrt(lx),np.sqrt(lx)]) 
		new_bounds = Interval(np.sqrt(lx),np.sqrt(lx))
		return MixedAffine(new_affine_form,new_bounds)

	a = lx
	b = ux

	fa = np.sqrt(a)
	fb = np.sqrt(b)

	alpha = (fb-fa)/(b-a)

	u =1./((2*alpha)*(2*alpha))

	r = lambda x : alpha*x+((-alpha*b)+(fb))

	dzeta = ((np.sqrt(u)+r(u))/2.0)-(alpha*u)

	delta = abs(np.sqrt(u)-r(u))/2.0

	alpha = mpmath.mpmathify(alpha)
	dzeta = mpmath.mpmathify(dzeta)
	delta = mpmath.mpmathify(delta)

	new_affine_form = self.affine_form._affineConstructor(alpha, dzeta, delta)
	new_bounds = Interval(np.sqrt(lx),np.sqrt(ux))
	return MixedAffine(new_affine_form,new_bounds)




def LogChebyshev_mixed(self):
	lx, ux = float(self.bounds.inf), float(self.bounds.sup)

	if lx==ux:
		new_affine_form = Affine([np.log(lx),np.log(lx)]) 
		new_bounds = Interval(np.log(lx),np.log(lx))
		return MixedAffine(new_affine_form,new_bounds)

	a = lx
	b = ux

	fa = np.log(a)
	fb = np.log(b)

	alpha = (fb-fa)/(b-a)

	u =1./alpha

	r = lambda x : alpha*x+((-alpha*b)+(fb))

	dzeta = ((np.log(u)+r(u))/2.0)-(alpha*u)

	delta = abs(np.log(u)-r(u))/2.0

	alpha = mpmath.mpmathify(alpha)
	dzeta = mpmath.mpmathify(dzeta)
	delta = mpmath.mpmathify(delta)

	new_affine_form = self.affine_form._affineConstructor(alpha, dzeta, delta)
	new_bounds = Interval(np.log(lx),np.log(ux))
	return MixedAffine(new_affine_form,new_bounds)




def StandardNormalCDFDeepZ_mixed(self):
	lx, ux = float(self.bounds.inf), float(self.bounds.sup)

	if lx==ux:
		new_affine_form = Affine([norm.cdf(lx),norm.cdf(lx)])
		new_bounds = Interval(norm.cdf(lx),norm.cdf(lx))
		return MixedAffine(new_affine_form,new_bounds)

	lambda_opt = min(norm.pdf(lx),norm.pdf(ux))

	mu1 = 0.5*(norm.cdf(ux)+norm.cdf(lx)-(lambda_opt*(ux+lx)))

	mu2 = 0.5*(norm.cdf(ux)-norm.cdf(lx)-(lambda_opt*(ux-lx)))

	alpha = mpmath.mpmathify(lambda_opt)
	dzeta = mpmath.mpmathify(mu1)
	delta = mpmath.mpmathify(mu2)

	new_affine_form = self.affine_form._affineConstructor(alpha, dzeta, delta)
	new_bounds = Interval(norm.cdf(lx),norm.cdf(ux))
	return MixedAffine(new_affine_form,new_bounds)



#Abstract transformer for the function f(x,y) = sigmoid'(x)*y
def SynthesizedSigmoidTransformer1Way_mixed(x1,y1,N=8):

	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [SigmoidPrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:
	sig = 0.00001
	A1 = A1 +  np.random.normal(0, sig)
	assert(A1 != 0.)

	#bound the error of the linear approximation
	pts_ly = InverseSigmoidDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseSigmoidDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(SigmoidPrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)

	corner_evals = [SigmoidPrimeProduct(lx,ly),SigmoidPrimeProduct(lx,uy),SigmoidPrimeProduct(ux,ly),SigmoidPrimeProduct(ux,uy)]

	#notice that the second argument is set to 0 since there is no linear coefficient since this is just for boxes/intervals not zonos
	box_pts_ly = InverseSigmoidDoublePrime(ly,0,lx,ux)
	box_pts_uy = InverseSigmoidDoublePrime(uy,0,lx,ux)

	if len(box_pts_ly)>0:
		for ly_rt in box_pts_ly:
			corner_evals.append(SigmoidPrimeProduct(ly_rt,ly))

	if len(box_pts_uy)>0:
		for uy_rt in box_pts_uy:
			corner_evals.append(SigmoidPrimeProduct(uy_rt,uy))

	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)


def SynthesizedTanhTransformer1Way_mixed(x1,y1,N=8):

	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [TanhPrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:

	sig = 0.00001
	A1 = A1 +  np.random.normal(0, sig)
	assert(A1 != 0.)

	#bound the error of the linear approximation
	pts_ly = InverseTanhDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseTanhDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	
	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(TanhPrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)
	"""


	maxval = -np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = (-TanhPrimeProduct(x_pt, y_pt) + (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	other_maxval = -np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val_ = (TanhPrimeProduct(x_pt, y_pt) - (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val_ > other_maxval:
				other_maxval = val_

	max_violation_above = other_maxval
	max_violation_below = maxval
	assert(max_violation_above>0)
	assert(max_violation_below>0)
	max_val = ((max_violation_above+max_violation_below)*0.5)
	C1 = (C1 - max_violation_below) + max_val
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,max_val)
	"""

	corner_evals = [TanhPrimeProduct(lx,ly),TanhPrimeProduct(lx,uy),TanhPrimeProduct(ux,ly),TanhPrimeProduct(ux,uy)]

	#notice that the second argument is set to 0 since there is no linear coefficient since this is just for boxes/intervals not zonos
	box_pts_ly = InverseTanhDoublePrime(ly,0,lx,ux)
	box_pts_uy = InverseTanhDoublePrime(uy,0,lx,ux)

	if len(box_pts_ly)>0:
		for ly_rt in box_pts_ly:
			corner_evals.append(TanhPrimeProduct(ly_rt,ly))

	if len(box_pts_uy)>0:
		for uy_rt in box_pts_uy:
			corner_evals.append(TanhPrimeProduct(uy_rt,uy))

	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)





def SynthesizedExpPrimeProductTransformer_mixed(x1,y1,N=8):
	ExpPrimeProd = lambda x,y : np.exp(x)*y
	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [np.exp(a[0])*a[1] for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_


	pts_ly = InverseExpDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseExpDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs((ExpPrimeProd(x_pt,y_pt)) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)

	corner_evals = [ExpPrimeProd(lx,ly),ExpPrimeProd(lx,uy),ExpPrimeProd(ux,ly),ExpPrimeProd(ux,uy)]

	#notice that the second argument is set to 0 since there is no linear coefficient since this is just for boxes/intervals not zonos
	box_pts_ly = InverseExpDoublePrime(ly,0,lx,ux)
	box_pts_uy = InverseExpDoublePrime(uy,0,lx,ux)

	if len(box_pts_ly)>0:
		for ly_rt in box_pts_ly:
			corner_evals.append(ExpPrimeProd(ly_rt,ly))

	if len(box_pts_uy)>0:
		for uy_rt in box_pts_uy:
			corner_evals.append(ExpPrimeProd(uy_rt,uy))


	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)




#synthesize a transformer for 1/x*y
def SynthesizedLogPrimeProductTransformer_mixed(x1,y1,N=50):
	logPrimeProd = lambda x,y : y/x
	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [(a[1]/a[0]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	if A1 == 0.:
		A1 += np.random.normal(0, 0.00000001)

	#check the conditions:
	assert(A1 != 0.)

	pts_ly = InverseLogDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseLogDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs((y_pt/x_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)

	corner_evals = [logPrimeProd(lx,ly),logPrimeProd(lx,uy),logPrimeProd(ux,ly),logPrimeProd(ux,uy)]

	#notice that the second argument is set to 0 since there is no linear coefficient since this is just for boxes/intervals not zonos
	box_pts_ly = InverseLogDoublePrime(ly,0,lx,ux)
	box_pts_uy = InverseLogDoublePrime(uy,0,lx,ux)

	if len(box_pts_ly)>0:
		for ly_rt in box_pts_ly:
			corner_evals.append(logPrimeProd(ly_rt,ly))

	if len(box_pts_uy)>0:
		for uy_rt in box_pts_uy:
			corner_evals.append(logPrimeProd(uy_rt,uy))

	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)




def SynthesizedSqrtPrimeProductTransformer_mixed(x1,y1,N=50):
	sqrtPrimeProd = lambda x,y : (0.5/np.sqrt(x))*y

	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [sqrtPrimeProd(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:
	assert(A1 != 0.)

	pts_ly = InverseSqrtDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseSqrtDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(sqrtPrimeProd(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)

	corner_evals = [sqrtPrimeProd(lx,ly),sqrtPrimeProd(lx,uy),sqrtPrimeProd(ux,ly),sqrtPrimeProd(ux,uy)]

	#notice that the second argument is set to 0 since there is no linear coefficient since this is just for boxes/intervals not zonos
	box_pts_ly = InverseSqrtDoublePrime(ly,0,lx,ux)
	box_pts_uy = InverseSqrtDoublePrime(uy,0,lx,ux)

	if len(box_pts_ly)>0:
		for ly_rt in box_pts_ly:
			corner_evals.append(sqrtPrimeProd(ly_rt,ly))

	if len(box_pts_uy)>0:
		for uy_rt in box_pts_uy:
			corner_evals.append(sqrtPrimeProd(uy_rt,uy))

	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)






def SynthesizedSinPrimeProductTransformer_mixed(x1,y1,N=50):
	sinPrimeProd = lambda x,y : (np.cos(x))*y

	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [sinPrimeProd(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:
	assert(A1 != 0.)

	pts_ly = InverseSinDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseSinDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(sinPrimeProd(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)
	return res



#synthesizes a transformer for x*y
def SynthesizedXYProductTransformer_mixed(x1,y1,N=50):
	XYProd = lambda x,y : (x)*y

	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	if ((ux==0) and (lx==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [XYProd(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_



	x_pts = [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(XYProd(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)

	corner_evals = [lx*ly,lx*uy,ux*ly,ux*uy]
	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)





#synthesize a transformer for Phi'(x)*y where Phi(x) is the normal CDF
def SynthesizedNormalCDFPrimeProductTransformer_mixed(x1,y1,N=50):
	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)


	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [NormalCDFPrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:
	assert(A1 != 0.)

	pts_ly = InverseNormalCDFDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseNormalCDFDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(  NormalCDFPrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)

	corner_evals = [NormalCDFPrimeProduct(lx,ly),NormalCDFPrimeProduct(lx,uy),NormalCDFPrimeProduct(ux,ly),NormalCDFPrimeProduct(ux,uy)]

	#notice that the second argument is set to 0 since there is no linear coefficient since this is just for boxes/intervals not zonos
	box_pts_ly = InverseNormalCDFDoublePrime(ly,0,lx,ux)
	box_pts_uy = InverseNormalCDFDoublePrime(uy,0,lx,ux)

	if len(box_pts_ly)>0:
		for ly_rt in box_pts_ly:
			corner_evals.append(NormalCDFPrimeProduct(ly_rt,ly))

	if len(box_pts_uy)>0:
		for uy_rt in box_pts_uy:
			corner_evals.append(NormalCDFPrimeProduct(uy_rt,uy))

	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)




#Abstract transformer for the function f(x,y) = 3x^2*y
def SynthesizedCubePrimeProductTransformer_mixed(x1,y1,N=8):
	CubePrimeProduct = lambda x,y : (3*x*x*y)

	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [CubePrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:

	sig = 0.00001
	A1 = A1 +  np.random.normal(0, sig)
	assert(A1 != 0.)

	#bound the error of the linear approximation
	pass
	pts_ly = InverseCubeDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseCubeDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(CubePrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)

	corner_evals = [CubePrimeProduct(lx,ly),CubePrimeProduct(lx,uy),CubePrimeProduct(ux,ly),CubePrimeProduct(ux,uy)]

	if ((lx<0) and (ux>0)):
		corner_evals.append(0)

	#notice that the second argument is set to 0 since there is no linear coefficient since this is just for boxes/intervals not zonos
	box_pts_ly = InverseCubeDoublePrime(ly,0,lx,ux)
	box_pts_uy = InverseCubeDoublePrime(uy,0,lx,ux)

	if len(box_pts_ly)>0:
		for ly_rt in box_pts_ly:
			corner_evals.append(CubePrimeProduct(ly_rt,ly))

	if len(box_pts_uy)>0:
		for uy_rt in box_pts_uy:
			corner_evals.append(CubePrimeProduct(uy_rt,uy))

	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)




def SynthesizedFourthPrimeProductTransformer_mixed(x1,y1,N=8):
	FourthPrimeProduct = lambda x,y : (4*x*x*x*y)

	#get grid points
	lx, ux = float(x1.bounds.inf), float(x1.bounds.sup)

	xs = np.linspace(lx,ux,N)

	ly, uy = float(y1.bounds.inf), float(y1.bounds.sup)

	if ((uy==0) and (ly==0)):
		new_affine_form = Affine([0, 0])
		return MixedAffine(new_affine_form)

	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [FourthPrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:

	sig = 0.00001
	A1 = A1 +  np.random.normal(0, sig)
	assert(A1 != 0.)

	#bound the error of the linear approximation
	pass
	pts_ly = InverseFourthDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseFourthDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	"""
	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(FourthPrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,maxval)
	"""

	maxval = -np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = (-FourthPrimeProduct(x_pt, y_pt) + (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	other_maxval = -np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val_ = (FourthPrimeProduct(x_pt, y_pt) - (A1 * x_pt + B1 * y_pt + C1))  # Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val_ > other_maxval:
				other_maxval = val_

	max_violation_above = other_maxval
	max_violation_below = maxval
	assert(max_violation_above>0)
	assert(max_violation_below>0)
	max_val = ((max_violation_above+max_violation_below)*0.5)
	C1 = (C1 - max_violation_below) + max_val
	res = (A1*x1.affine_form)+(B1*y1.affine_form)+C1
	res = AddNewNoiseSymbol(res,max_val)



	corner_evals = [FourthPrimeProduct(lx,ly),FourthPrimeProduct(lx,uy),FourthPrimeProduct(ux,ly),FourthPrimeProduct(ux,uy)]

	if ((lx<0) and (ux>0)):
		corner_evals.append(0)

	#notice that the second argument is set to 0 since there is no linear coefficient since this is just for boxes/intervals not zonos
	box_pts_ly = InverseFourthDoublePrime(ly,0,lx,ux)
	box_pts_uy = InverseFourthDoublePrime(uy,0,lx,ux)

	if len(box_pts_ly)>0:
		for ly_rt in box_pts_ly:
			corner_evals.append(FourthPrimeProduct(ly_rt,ly))

	if len(box_pts_uy)>0:
		for uy_rt in box_pts_uy:
			corner_evals.append(FourthPrimeProduct(uy_rt,uy))

	corner_lower = min(corner_evals)
	corner_upper = max(corner_evals)
	refined_interval = Interval(corner_lower,corner_upper)
	return MixedAffine(res,refined_interval)




"""
def IntervalBoundTransformer1Way(x1,y1):
	#get grid points
	lx, ux = float(x1.interval.inf), float(x1.interval.sup)

	ly, uy = float(y1.interval.inf), float(y1.interval.sup)

	y_pts = [ly,uy]
	x_pts = [lx,ux]


	#if the [lx,ux] interval contains zero, we need to check the root x=0
	if not((lx>0) or (ux<0)):
		x_pts.append(0)


	maxval =-np.inf
	minval = np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = SigmoidPrimeProduct(x_pt,y_pt) #Evaluate the whole expression f'(x)y
			if val > maxval:
				maxval = val
			if val < minval:
				minval = val

	return [minval,maxval]



#Abstract transformer for the function f(x,y) = sigmoid'(x)*y
def RefinedSynthesizedSigmoidTransformer1Way(x1,y1,xs,ys,N=50):

	#get grid points
	lx, ux = max(float(x1.interval.inf),xs[0]), min(float(x1.interval.sup),xs[1])

	xs = np.linspace(lx,ux,N)

	ly, uy = max(float(y1.interval.inf),ys[0]), min(float(y1.interval.sup),ys[1])
	ys = np.linspace(ly,uy,N)

	xys = np.asarray(cartesian_product(xs,ys))


	#evaluate the true function (what we are overapproximating) on the grid points
	zs = [SigmoidPrimeProduct(a[0],a[1]) for a in xys]
	zs = np.asarray(zs)
	#print(xys)
	#print(zs)

	#perform linear regression on the grid points to get the coefficients
	reg = LinearRegression().fit(xys, zs)
	A1,B1 = reg.coef_
	C1 = reg.intercept_

	#check the conditions:
	assert(A1 != 0.)

	#bound the error of the linear approximation
	pass
	pts_ly = InverseSigmoidDoublePrime(ly,A1,lx,ux)
	pts_uy = InverseSigmoidDoublePrime(uy,A1,lx,ux)

	x_pts = pts_ly + pts_uy + [lx,ux]
	y_pts = [ly,uy]

	maxval =-np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = abs(SigmoidPrimeProduct(x_pt,y_pt) - (A1*x_pt+B1*y_pt+C1)) #Evaluate the whole expression |f'(x)y-(Ax+By+C)|
			if val > maxval:
				maxval = val

	#print(maxval)
	res = (A1*x1)+(B1*y1)+C1
	res = AddNewNoiseSymbol(res,maxval)
	return res


def RefinedIntervalBoundTransformer1Way(x1,y1,xs,ys):
	#get grid points
	lx, ux = max(float(x1.interval.inf),xs[0]), min(float(x1.interval.sup),xs[1])

	ly, uy = max(float(y1.interval.inf),ys[0]), min(float(y1.interval.sup),ys[1])

	y_pts = [ly,uy]
	x_pts = [lx,ux]


	#if the [lx,ux] interval contains zero, we need to check the root x=0
	if not((lx>0) or (ux<0)):
		x_pts.append(0)


	maxval =-np.inf
	minval = np.inf
	for x_pt in x_pts:
		for y_pt in y_pts:
			val = SigmoidPrimeProduct(x_pt,y_pt) #Evaluate the whole expression f'(x)y
			if val > maxval:
				maxval = val
			if val < minval:
				minval = val

	return [minval,maxval]
"""


