import numpy as np

def f_orig(x):
	return (-np.exp(-0.5*(x*x))*x)/np.sqrt(2.0*np.pi)

def f_prime(x):
	return np.exp(-0.5*(x*x))*((x*x)-1)*(1.0/np.sqrt(2.0*np.pi))


def bisect():
	pass

#break things up into the monotonic intervals and then do logarithmic 
#number of hi/lo searches
def search_neg_inf_to_minus1(y):
	f_shifted = lambda x : f_orig(x)-y
	res = my_newton(f_shifted,f_prime,-2,1e-10)
	assert(res<-1)
	return res

def search_minus1_to_0(y):
	f_shifted = lambda x : f_orig(x)-y
	res = my_newton(f_shifted,f_prime,-0.5,1e-10)
	assert((res>-1) and (res<0))
	return res


def search_0_to_1(y):
	f_shifted = lambda x : f_orig(x)-y
	res = my_newton(f_shifted,f_prime,0.5,1e-10)
	assert((res>0) and (res<1))
	return res

def search_1_to_inf(y):
	f_shifted = lambda x : f_orig(x)-y
	res = my_newton(f_shifted,f_prime,2,1e-10)
	assert((res>1))
	return res



#https://pythonnumericalmethods.berkeley.edu/notebooks/chapter19.04-Newton-Raphson-Method.html
def my_newton(f, df, x0, tol):
    # output is an estimation of the root of f 
    # using the Newton Raphson method
    # recursive implementation
    if abs(f(x0)) < tol:
        return x0
    else:
        return my_newton(f, df, x0 - f(x0)/df(x0), tol)

def invert_erf_2nd_deriv(y):
#	c = 1./np.sqrt(np.exp(1))
	c = 1./(np.sqrt(2*(np.exp(1)*np.pi)))
	if (abs(y) > c):
		return (float('NaN'),float('NaN')) #no inverse found

	elif y==c:
		return (float(-1),float('NaN'))

	elif y==-c:
		return (float(1),float('NaN'))

	elif y==0:
		return (0,float('NaN'))

	else:
		if ((0 < y) and (y < c)):
			r1 = search_neg_inf_to_minus1(y)
			r2 = search_minus1_to_0(y)

		elif ((0 > y) and (y > -c)):
			r1 = search_0_to_1(y)
			r2 = search_1_to_inf(y)
		else:
			raise Exception

		return (r1,r2)
