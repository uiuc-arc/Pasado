import sys
sys.path.insert(1, '../')
from synthesized_transformer_mixed import *
from dual_intervals import *


def test1():
	a = Affine(Interval(-2,0))
#	refined = Interval(-1.5,0)
#	a_ = MixedAffine(a,refined)
	a_ = MixedAffine(a)
	b = a_ + 3.

	c = ExpChebyshev_mixed(b)
	print(c.bounds)
	print(c.affine_form.interval)
	

def test_synthesized():
	a = Affine(Interval(-2,0))
#	refined = Interval(-1.5,0)
#	a_ = MixedAffine(a,refined)
	a_ = MixedAffine(a)
	b = a_ + 3.

	c = SynthesizedExpPrimeProductTransformer_mixed(a_,b)
	print(c.bounds)
	print(c.affine_form.interval)


def test_synthesized_product():
	a_ = Affine(Interval(-2,-1))
	refined = Interval(-1.5,-.5)
#	a = MixedAffine(a_)
	a = MixedAffine(a_,refined)

	b_ = Affine(Interval(5,6))
	b = MixedAffine(b_)


	c_ = Affine(Interval(7.9,9))
	c = MixedAffine(c_)

	d_ = Affine(Interval(2.5,3.2))
	d = MixedAffine(d_)


	ours = SynthesizedProductRuleTransformer_mixed(a,b,c,d)
	print(ours.bounds)
	baseline = (a*d)+(b*c)
	print(baseline.bounds)

	ours_without = SynthesizedProductRuleTransformer(a_,b_,c_,d_)
#	print(ours_without.interval)
	baseline_without = (a_*d_)+(b_*c_)
#	print(baseline_without.interval)	
	print(ours.affine_form)
	print(baseline.affine_form)



def test_synthesized_quotient():
	a_ = Affine(Interval(-2,-1))
#	refined = Interval(-1.5,-.5)
	a = MixedAffine(a_)
#	a = MixedAffine(a_,refined)

	b_ = Affine(Interval(5,6))
	b = MixedAffine(b_)


	c_ = Affine(Interval(7.9,9))
	c = MixedAffine(c_)

	d_ = Affine(Interval(2.5,3.2))
	d = MixedAffine(d_)

	ours_without = SynthesizedQuotientRuleTransformer(a_,b_,c_,d_)
	print(ours_without.interval)
	ours = SynthesizedQuotientRuleTransformer_mixed(a,b,c,d)
	print(ours.bounds)
	baseline = ((b*c)-(a*d))/(c*c)#(a*d)+(b*c)
	print(baseline.bounds)
	baseline_without = ((b_*c_)-(a_*d_))/(c_*c_)#(a*d)+(b*c)
	print(baseline_without.interval)


	print("\n")

#	print(ours.affine_form)
#	print(baseline.affine_form)


	vanilla_interval = ((b_.interval*c_.interval)-(a_.interval*d_.interval))/(c_.interval*c_.interval)
	print(vanilla_interval)


def test_mixedaffine_dual_log():
	real = MixedAffine(Affine(Interval(0.5,10)),Interval(1,2))
	dual = MixedAffine(Affine(Interval(1, 1)))

	D1 = Dual(real,dual)
	cc = LogDual(D1)
#	cc = NormalCDFDual(D1)

	print(cc.dual.bounds)
	print(cc.dual.affine_form.interval)


def test_mixedaffine_dual_log_precise():
	real = MixedAffine(Affine(Interval(0.5,10)),Interval(1,2))
	dual = MixedAffine(Affine(Interval(1, 1)))

	D1 = Dual(real,dual)
	cc = LogDualPrecise(D1)
#	cc = NormalCDFDualPrecise(D1)
	print(cc.dual.bounds)
	print(cc.dual.affine_form.interval)


def test_mixedaffine_dual_sqrt():
	real = MixedAffine(Affine(Interval(0.5,10)),Interval(1,2))
	dual = MixedAffine(Affine(Interval(1, 1)))

	D1 = Dual(real,dual)
	cc = SqrtDual(D1)
#	cc = NormalCDFDual(D1)

	print(cc.dual.bounds)
	print(cc.dual.affine_form.interval)


def test_mixedaffine_dual_sqrt_precise():
	real = MixedAffine(Affine(Interval(0.5,10)),Interval(1,2))
	dual = MixedAffine(Affine(Interval(1, 1)))

	D1 = Dual(real,dual)
	cc = SqrtDualPrecise(D1)
#	cc = NormalCDFDualPrecise(D1)
	print(cc.dual.bounds)
	print(cc.dual.affine_form.interval)




def test_interval_dual_division_():
	real = ((Interval(5,12)))
	dual = ((Interval(0, 0)))
	D1 = Dual(real,dual)


	real2 = ((Interval(2,3)))
	dual2 = ((Interval(1, 1)))
	D2 = Dual(real2,dual2)

	cc = D1/D2
#	cc = (cc/D2)
	cc = SigmoidDual(cc)
	cc = cc * cc
#	print(cc.real.interval)
	print("Box")
	print(cc.dual)
	print("\n")
#	print(cc.dual)

#	f = ExpDual(cc)
#	print(f.dual)



def test_affine_dual_division():
	real = (Affine(Interval(5,12)))
	dual = (Affine(Interval(0, 0)))
	D1 = Dual(real,dual)


	real2 = (Affine(Interval(2,3)))
	dual2 = (Affine(Interval(1, 1)))
	D2 = Dual(real2,dual2)

	cc = D1/D2
#	cc = (cc/D2)
	cc = SigmoidDual(cc)
	cc = cc * cc
#	print(cc.real.interval)
#	print(cc.dual.affine_form.interval)
	print("vanilla zonos")
	print(cc.dual.interval)
	print("\n")
#	f = ExpDual(cc)
#	print(f.dual.interval)


def test_mixedaffine_dual_division():
	real = MixedAffine(Affine(Interval(5,12)))
	dual = MixedAffine(Affine(Interval(0, 0)))
	D1 = Dual(real,dual)


	real2 = MixedAffine(Affine(Interval(2,3)))
	dual2 = MixedAffine(Affine(Interval(1, 1)))
	D2 = Dual(real2,dual2)

	cc = D1/D2
#	cc = (cc/D2)
	cc = SigmoidDual(cc)
	cc = cc * cc
#	cc = D1*D2
#	print(cc.real.interval)
	print("Refined vanilla zonos")
	print(cc.dual.affine_form.interval)
	print(cc.dual.bounds)
	print("\n")
#	f = ExpDual(cc)
#	print(f.dual.bounds)



def test_mixedaffine_dual_division_precise():
	real = MixedAffine(Affine(Interval(5,12)))
	dual = MixedAffine(Affine(Interval(0, 0)))
	D1 = Dual(real,dual)


	real2 = MixedAffine(Affine(Interval(2,3)))
	dual2 = MixedAffine(Affine(Interval(1, 1)))
	D2 = Dual(real2,dual2)

	cc = DividePrecise(D1,D2)
#	cc = (DividePrecise(cc,D2))
	cc = SigmoidDualPrecise(cc)
	cc = MultiplyPrecise(cc,cc)
#	print(cc.real.bounds)
	print("Refined AND Synthesized zonos (our best)")
	print(cc.dual.affine_form.interval)
	print(cc.dual.bounds)
	print("\n")

#	f = ExpDualPrecise(cc)
#	print(f.dual.bounds)



def test_affine_dual_division_precise():
	real = (Affine(Interval(5,12)))
	dual = (Affine(Interval(0, 0)))
	D1 = Dual(real,dual)


	real2 = (Affine(Interval(2,3)))
	dual2 = (Affine(Interval(1, 1)))
	D2 = Dual(real2,dual2)

	cc = DividePrecise(D1,D2)
#	cc = (DividePrecise(cc,D2))
	cc = SigmoidDualPrecise(cc)
	cc = MultiplyPrecise(cc,cc)
#	print(cc.real.interval)
#	print(cc.dual.affine_form.interval)
	print("Synthesized zonos")
	print(cc.dual.interval)
	print("\n")

#	f = ExpDualPrecise(cc)
#	print(f.dual.interval)


#test1()
#test_synthesized_product()
#test_synthesized_quotient()

#test_mixedaffine_dual_log()
#test_mixedaffine_dual_log_precise()


#test_mixedaffine_dual_sqrt()
#test_mixedaffine_dual_sqrt_precise()

test_interval_dual_division_()
test_affine_dual_division()
test_affine_dual_division_precise()
test_mixedaffine_dual_division()
test_mixedaffine_dual_division_precise()

