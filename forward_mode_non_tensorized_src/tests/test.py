import sys
sys.path.insert(1, '../')
from dual_intervals import *



def test_affine_dual_sub():
	real = Affine(Interval(0,1))
	dual = Affine(Interval(1, 1.01))

	D1 = Dual(real,dual)
	cc = D1-D1#ExpDual(D1)/D1*2
	#cc = ExpDual(cc)
	print(cc.dual.interval)
	#print(cc.real.interval)


def test_affine_dual_log():
	real = Affine(Interval(0.5,1))
	dual = Affine(Interval(1, 2))

	D1 = Dual(real,dual)
	cc = LogDual(D1)

	print(cc.dual.interval)


def test_affine_dual_log_precise():
	real = Affine(Interval(0.5,1))
	dual = Affine(Interval(1, 2))

	D1 = Dual(real,dual)
	cc = LogDualPrecise(D1)

	print(cc.dual.interval)







def IMR_feasnewt(x0,x1,x2,y0,y1,y2):
	x3 = y0
	x4 = y1
	x5 = y2
	sqrt3 = np.sqrt(3.)
	m0 = x1-x0
	
	m1 = (2*x2-x1-x0)*sqrt3

	m2 = x4-x3 
	m3 = (2*x5-x4-x3)*sqrt3

	g = (m0*m3) - (m1*m2)

	f = SquareDual(m0)+SquareDual(m1)+SquareDual(m2)+SquareDual(m3)

	obj = 0.5 * (f/g)
	return obj


def test_IMR():
	x0,y0 = 0,0
	x1,y1 = 1,1
	x2,y2 = 2,0
	x0 = Dual(x0,0)
	x1 = Dual(x1,0)
	x2 = Dual(x2,0)
	y0 = Dual(y0,0)
	y1 = Dual(y1,0)
	y2 = Dual(y2,1)
	obj = IMR_feasnewt(x0,x1,x2,y0,y1,y2)


def black_scholes():
	pass


def test_affine_dual_sigmoid():
	real = Affine(Interval(-0.8,0.8))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = SigmoidDual(D1)
	print(cc.real.interval)
	print(cc.dual.interval)



def test_affine_dual_sigmoid_precise():
	real = Affine(Interval(-0.,0.8))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = SigmoidDualPrecise(D1)
	print(cc.real.interval)
	print(cc.dual.interval)

def test_affine_dual_tanh():
	real = Affine(Interval(-0.8,0.8))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = TanhDual(D1)
	#print(cc.real.interval)
	print(cc.dual.interval)

def test_affine_dual_tanh_precise():
	real = Affine(Interval(-0.8,0.8))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = TanhDualPrecise(D1)
	#print(cc.real.interval)
	print(cc.dual.interval)

def test_mixedaffine_dual_tanh():
	real = MixedAffine(Affine(Interval(-0.8,0.8)))
	dual = MixedAffine(Affine(Interval(1, 1)))

	D1 = Dual(real,dual)
	cc = TanhDual(D1)
	#print(cc.real.bounds)
	print(cc.dual.bounds)


def test_mixedaffine_dual_tanh_precise():
	real = MixedAffine(Affine(Interval(-0.8,0.8)))
	dual = MixedAffine(Affine(Interval(1, 1)))

	D1 = Dual(real,dual)
	cc = TanhDualPrecise(D1)
	#print(cc.real.bounds)
	print(cc.dual.affine_form.interval)
	print(cc.dual.bounds)






def test_quotient_rule_transformer():
	a = Affine([10,10])
	b = Affine([1,1.2])
	c = Affine([1,1.1])
	d = Affine([1,1])
	z = SynthesizedQuotientRuleTransformer(a,b,c,d)
	print(z.interval)

	f = ((b*c)-(a*d))/(c.sqr())
	print(f.interval)



def test_interval_dual_division():
	real = (Interval(-5,12))
	dual = (Interval(1, 1))
	D1 = Dual(real,dual)


	real2 = (Interval(5,7))
	dual2 = (Interval(0, 1))
	D2 = Dual(real2,dual2)

	cc = D1/D2
#	print(cc.real)
#	print(cc.dual.interval)
	print(cc.dual)



def test_affine_dual_division():
	real = Affine(Interval(-5,12))
	dual = Affine(Interval(1, 1))
	D1 = Dual(real,dual)


	real2 = Affine(Interval(5,7))
	dual2 = Affine(Interval(0, 1))
	D2 = Dual(real2,dual2)

	cc = D1/D2
#	print(cc.real.interval)
	print(cc.dual.interval)
#	print(cc.dual)


def test_affine_dual_division_precise():
	real = Affine(Interval(-5,12))
	dual = Affine(Interval(1, 1))
	D1 = Dual(real,dual)


	real2 = Affine(Interval(5,7))
	dual2 = Affine(Interval(0, 1))
	D2 = Dual(real2,dual2)

	cc = DividePrecise(D1,D2)
#	print(cc.real.interval)
	print(cc.dual.interval)
#	print(cc.dual)



def test_affine_dual_normalcdf():
	real = Affine(Interval(0.5,1))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = NormalCDFDual(D1)
	print(cc.real.interval)
	print(cc.dual.interval)


def test_affine_dual_exp():
	real = Affine(Interval(-.5,4))
	dual = Affine(Interval(0.5, 4))

	D1 = Dual(real,dual)
	cc = ExpDual(D1)

	print(cc.dual.interval)


def test_affine_dual_exp_precise():
	real = Affine(Interval(-.5,4))
	dual = Affine(Interval(0.5, 4))

	D1 = Dual(real,dual)
	cc = ExpDualPrecise(D1)

	print(cc.dual.interval)




def test_affine_dual_sqrt():
	real = Affine(Interval(0.2,5))
	dual = Affine(Interval(1, 3))

	D1 = Dual(real,dual)
	cc = SqrtDual(D1)

#	print(cc.real.interval)
	print(cc.dual.interval)


def test_affine_dual_sqrt_precise():
	real = Affine(Interval(0.2,5))
	dual = Affine(Interval(1, 3))

	D1 = Dual(real,dual)
	cc = SqrtDualPrecise(D1)

#	print(cc.real.interval)
	print(cc.dual.interval)





def test_affine_dual_multiplication():
	real = Affine(Interval(0.5,10))
#	real = (Interval(0.5,10))
	dual = Affine(Interval(-1, 12))
#	dual = (Interval(-1, 12))
	D1 = Dual(real,dual)


	real2 = Affine(Interval(5,70))
	dual2 = Affine(Interval(2,3))
#	real2 = (Interval(5,70))
#	dual2 = (Interval(2,3))
	D2 = Dual(real2,dual2)

	cc = D1*D2
	print(cc.real.interval)
	print(cc.real)
	print("\n")
	print(cc.dual.interval)
	print(cc.dual)


def test_affine_dual_multiplication_precise():
	real = Affine(Interval(0.5,10))
	dual = Affine(Interval(-1, 12))
	D1 = Dual(real,dual)


	real2 = Affine(Interval(5,70))
	dual2 = Affine(Interval(2, 3))
	D2 = Dual(real2,dual2)

	cc = MultiplyPrecise(D1,D2)
	print(cc.real.interval)
	print(cc.real)
	print("\n")
	print(cc.dual.interval)
	print(cc.dual)



def test_invert_gaussian_cdf():
	a,b = inverse_erf_2nd_deriv(0.1)
	print(a)
	print(b)
	print(ig.f_orig(a))
	print(ig.f_orig(b))






def test_affine_dual_normal_cdf():
	real = Affine(Interval(0,1))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = NormalCDFDual(D1)

	print(cc.dual.interval)
	print(cc.dual)


def test_affine_dual_normal_cdf_precise():
	real = Affine(Interval(0,1))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = NormalCDFDualPrecise(D1)

	print(cc.dual.interval)
	print(cc.dual)

def test_inv_sin():
	a = 0.3
	
	l = -1
	u = 1+6.5
	
	pts = InverseSinDoublePrime(1,a,l,u)
	print(pts)
	



def test_affine_dual_sin():
	real = Affine(Interval(-1,2))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = SinDual(D1)

	print(cc.dual.interval)
#	print(cc.dual)


def test_affine_dual_sin_precise():
	real = Affine(Interval(-1,2))
	dual = Affine(Interval(1, 1))

	D1 = Dual(real,dual)
	cc = SinDualPrecise(D1)

	print(cc.dual.interval)
#	print(cc.dual)



def test_synthesized_xy_product_precise():
	x = Affine(Interval(-10,2))
	y = Affine(Interval(1,22))

	z = SynthesizedXYProductTransformer(x,y)
	print(z.interval)

	z2 = x*y
	print(z2.interval)

	z3 = x.interval*y.interval
	print(z3)



def test_affine_dual_cube():
	real = Affine(Interval(0.2,5))
	dual = Affine(Interval(1, 3))

	D1 = Dual(real,dual)
	cc = CubeDual(D1)

#	print(cc.real.interval)
	print(cc.dual.interval)


def test_affine_dual_cube_precise():
	real = Affine(Interval(0.2,5))
	dual = Affine(Interval(1, 3))

	D1 = Dual(real,dual)
	cc = CubeDualPrecise(D1)

#	print(cc.real.interval)
	print(cc.dual.interval)







test_affine_dual_tanh()
#test_affine_dual_tanh_precise()
#test_mixedaffine_dual_tanh()
test_mixedaffine_dual_tanh_precise()

test_affine_dual_sin()
test_affine_dual_sin_precise()

#test_inv_sin()

#test_affine_dual_sqrt()
#test_affine_dual_sqrt_precise()

#test_affine_dual_normal_cdf()
#test_affine_dual_normal_cdf_precise()

#test_invert_gaussian_cdf()

#test_affine_dual_multiplication()
#test_affine_dual_multiplication_precise()

#test_affine_dual_log()
#test_affine_dual_log_precise()


#test_affine_dual_exp()
#test_affine_dual_exp_precise()

#test_stencil()
#test_IMR()

#test_affine_dual_sigmoid()
#test_affine_dual_sigmoid_precise()

#test_interval_dual_division()
#test_affine_dual_division()
#test_affine_dual_division_precise()

#test_affine_dual_normalcdf()

#test_synthesized_xy_product_precise()

#test_affine_dual_cube()
#test_affine_dual_cube_precise()
