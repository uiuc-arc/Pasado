import sys
sys.path.insert(1, '../')
from runge_kutta import *

#https://kitchingroup.cheme.cmu.edu/blog/2018/10/11/A-differentiable-ODE-integrator-for-sensitivity-analysis/
def example():
#	k1 = 3.0

	k1_real = Affine(Interval(2.5,3.5))
	k1_dual = Affine(Interval(0, 0))
	k1 = Dual(k1_real,k1_dual)


	k_1_real = Affine(Interval(2,4))
	k_1_dual = Affine(Interval(1,1))
	k_1 = Dual(k_1_real,k_1_dual)



#	k_1 = 3.0
	Ca0 = 1.0
#	Ca0_real = Affine(Interval(.95,1.05))
#	Ca0_dual = Affine(Interval(0, 0))
#	Ca0 = Dual(Ca0_real,Ca0_dual)


	func = lambda t,Ca : (Ca*(-k1) ) + ( k_1 * (Ca0 + (-Ca) ))
#	func = lambda t,Ca : MultiplyPrecise(Ca,(-k1)) + MultiplyPrecise( k_1 , (Ca0 + (-Ca) ))


	eps = 0.2
	y0_real = Affine(Interval(Ca0-eps,Ca0+eps))
	y0_dual = Affine(Interval(0, 0))
	y0 = Dual(y0_real,y0_dual)


	t0_real = Affine(Interval(0,0))
	t0_dual = Affine(Interval(0, 0))
	t0 = Dual(t0_real,t0_dual)


	ys = [y0]
	ts = [t0]
	for i in range(60):
		y_new,t_new = runge_kutta(ys[-1],ts[-1],func,0.01)
		ys.append(y_new)
		ts.append(t_new)
		
#	print(ys[-1].real)	
#	print(ys[-1].dual)

#	print([x.real for x in ys])
	[print(x.dual.interval) for x in ys]

#	plot_derivs(ts,ys)






#https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
def test():

	y0_real = Affine(Interval(0.5,0.55))
	y0_dual = Affine(Interval(1, 1))
	y0 = Dual(y0_real,y0_dual)


	t0_real = Affine(Interval(0,0))
	t0_dual = Affine(Interval(0, 0))
	t0 = Dual(t0_real,t0_dual)

	func = lambda t,x : (x - (t*t)) + 1

	#y1,t1 = runge_kutta(y0,t0,func,0.5)

#	print(y1.real.interval)
#	print(y1.dual.interval)
#	print(y1.real.interval)
#	print(t1.real.interval)

	ys = [y0]
	ts = [t0]
	for i in range(4):

		y_new,t_new = runge_kutta(ys[-1],ts[-1],func,0.5)
		ys.append(y_new)
		ts.append(t_new)
		
#	print([x.real.interval for x in ys])	
	print([x.dual.interval for x in ys])	




#https://www.intmath.com/differential-equations/12-runge-kutta-rk4-des.php
def test2():

	y0_real = Affine(Interval(.9,1.2))
	y0_dual = Affine(Interval(1, 1))
	y0 = Dual(y0_real,y0_dual)


#	beta_real = Affine(Interval(-.2,.3))
#	beta_dual = Affine(Interval(1, 1))
#	beta = Dual(beta_real,beta_dual)


	t0_real = Affine(Interval(0,0))
	t0_dual = Affine(Interval(0, 0))
	t0 = Dual(t0_real,t0_dual)

#	func = lambda t,x : (5*SquareDual(t)-x)/(ExpDual(t+x)) 
	func = lambda t,x : DividePrecise((5*MultiplyPrecise(t,t)-x),(ExpDualPrecise(t+x))) #(x - (t*t)) + 1

	ys = [y0]
	ts = [t0]
	for i in range(10):

		y_new,t_new = runge_kutta(ys[-1],ts[-1],func,0.1)
		ys.append(y_new)
		ts.append(t_new)
		
#	print([x.real.interval for x in ys])	
#	print("\n")
	print([x.dual.interval for x in ys])	




#http://dimacs.rutgers.edu/archive/MPE/Energy/DIMACS-EBM.pdf
#https://www.e-education.psu.edu/meteo469/node/137
def climate():


	eps_green_real = Affine(Interval(0.5,0.7))
	eps_green_dual = Affine(Interval(1,1))
	eps_green = Dual(eps_green_real,eps_green_dual)


	sigma = 5.67e-8
	S = 1367.6
	Q = 0.25*S

	C = 2.08e8

	alpha_f = lambda z : -(0.4*SigmoidDualPrecise((z-265)*0.2))+0.7

	func = lambda t,T : (((-alpha_f(T)+1.)*Q)-(eps_green*sigma*SquareDual(SquareDual(T))))/C




	T0_real = Affine(Interval(286,288))
	T0_dual = Affine(Interval(0, 0))
	T0 = Dual(T0_real,T0_dual)


	t0_real = Affine(Interval(0,0))
	t0_dual = Affine(Interval(0, 0))
	t0 = Dual(t0_real,t0_dual)


	ys = [T0]
	ts = [t0]
	for i in range(10):
		y_new,t_new = runge_kutta(ys[-1],ts[-1],func,10)
		ys.append(y_new)
		ts.append(t_new)
		
#	print(ys[-1].real)	
#	print(ys[-1].dual)

#	print([x.real.interval for x in ys])
	[print(x.dual.interval) for x in ys]


#https://math.libretexts.org/Bookshelves/Differential_Equations/Elementary_Differential_Equations_with_Boundary_Value_Problems_(Trench)/01%3A_Introduction/1.01%3A_Applications_Leading_to_Differential_Equations
def glucose():

	y0_real = Affine(Interval(-2,2))
	y0_dual = Affine(Interval(0, 0))
	y0 = Dual(y0_real,y0_dual)

	L_real = Affine(Interval(.5,5.))
	L_dual = Affine(Interval(1, 1))
	L = Dual(L_real,L_dual)


	t0_real = Affine(Interval(0,0))
	t0_dual = Affine(Interval(0, 0))
	t0 = Dual(t0_real,t0_dual)

	func = lambda t,x : -L*ExpDual((-t*L))  
#	func = lambda t,x : MultiplyPrecise(-L,(ExpDualPrecise(MultiplyPrecise(t,-L))))

	ys = [y0]
	ts = [t0]
	for i in range(10):

		y_new,t_new = runge_kutta(ys[-1],ts[-1],func,0.1)
		ys.append(y_new)
		ts.append(t_new)
		
#	print([x.real.interval for x in ys])	
#	print("\n")
	[print(x.dual.interval) for x in ys]


def ricatti():

	y0_real = Affine(Interval(-1,-0.9))
	y0_dual = Affine(Interval(0, 0))
	y0 = Dual(y0_real,y0_dual)


	D_real = Affine(Interval(0.12,.13))
	D_dual = Affine(Interval(1, 1))
	D = Dual(D_real,D_dual)


	delta_real = Affine(Interval(.2,.3))
	delta_dual = Affine(Interval(0, 0))
	delta = Dual(delta_real,delta_dual)


	t0_real = Affine(Interval(0,0))
	t0_dual = Affine(Interval(0, 0))
	t0 = Dual(t0_real,t0_dual)

#	Beta = lambda t : 0.5*(ExpDual(-delta*t)+1.)
	Beta = lambda t : -(D/(delta*t-1))+1
#	Beta = lambda t : 0.5*(ExpDualPrecise(MultiplyPrecise(-delta,t))+1.)


#	func = lambda t,y : y + 2.01*(Beta(t)*SquareDual(y))
	func = lambda t,y : y + 2.01*MultiplyPrecise(Beta(t),SquareDual(y))


	ys = [y0]
	ts = [t0]
	for i in range(6):

		y_new,t_new = runge_kutta(ys[-1],ts[-1],func,0.5)
		ys.append(y_new)
		ts.append(t_new)
		
#	print([x.real.interval for x in ys])	
#	print("\n")
	[print(x.dual.interval) for x in ys]



#https://www.intmath.com/differential-equations/12-runge-kutta-rk4-des.php
def test_simple_neural():

	y0_real = Affine(Interval(.7,1.25))
	y0_dual = Affine(Interval(0, 0))
	y0 = Dual(y0_real,y0_dual)


	beta_real = Affine(Interval(-.5,.52))
	beta_dual = Affine(Interval(1, 1))
	beta = Dual(beta_real,beta_dual)


	t0_real = Affine(Interval(0,0))
	t0_dual = Affine(Interval(0, 0))
	t0 = Dual(t0_real,t0_dual)

#	func = lambda t,x : (5*SquareDual(t)-x)/(ExpDual(t+x)) 
	func = lambda t,x : SigmoidDual(t)*SigmoidDual(3*SigmoidDual(beta*x)+-2*SigmoidDual(1.4*x)) #(x - (t*t)) + 1
#	func = lambda t,x : SigmoidDualPrecise(t)*SigmoidDualPrecise(3*SigmoidDualPrecise(MultiplyPrecise(beta,x))+-2*SigmoidDualPrecise(1.4*x)) #(x - (t*t)) + 1

	ys = [y0]
	ts = [t0]
	for i in range(10):

		y_new,t_new = runge_kutta(ys[-1],ts[-1],func,0.1)
		ys.append(y_new)
		ts.append(t_new)
		
#	print([x.real.interval for x in ys])	
#	print("\n")
	[print(x.dual.interval) for x in ys]


#test_simple_neural()

#test2()
test()
#glucose()
#ricatti()
#example()
climate()
