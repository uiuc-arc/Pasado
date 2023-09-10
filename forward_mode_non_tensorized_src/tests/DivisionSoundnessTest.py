import sys
sys.path.insert(1, '../')
from dual_intervals import *

#This is just to check the soundness of everything for the Division rule
def affine_precise():
    size = 1000000

    X1_l = 80
    X1_u = 110
    X1 = MixedAffine(Affine(Interval(X1_l,X1_u)))
    X1s = np.random.uniform(X1_l,X1_u,size)


    Y1_l = -1
    Y1_u = 200
    Y1 = MixedAffine(Affine(Interval(Y1_l, Y1_u)))
    Y1s = np.random.uniform(Y1_l,Y1_u,size)
    a = Dual(X1, Y1)

    X2_l = 1.5
    X2_u = 1.5510010
    X2 = MixedAffine(Affine(Interval(X2_l,X2_u)))
    X2s = np.random.uniform(X2_l,X2_u,size)

    Y2_l = -21
    Y2_u = 208
    Y2 = MixedAffine(Affine(Interval(Y2_l, Y2_u)))
    Y2s = np.random.uniform(Y2_l,Y2_u,size)
    b = Dual(X2, Y2)


    res = DividePrecise(a, b)
    
    final = res.dual.bounds
    

    quotient_rule = ((Y1s*X2s)-(X1s*Y2s))/(X2s*X2s)


    dlow, dup = final.inf,final.sup
    assert((quotient_rule<=dup).all())
    assert((quotient_rule>=dlow).all())
#    assert(quotient_rule.all() <= dup)


#    assert(quotient_rule.all() >= dlow)

    print(min(quotient_rule))
    print(max(quotient_rule))    
    print(final)

    return final

affine_precise()
