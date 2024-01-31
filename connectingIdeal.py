import itertools
from sage.all import *
from helpers import ReducedBasis, QuaternionOrderBasis, isIsomorphic, IsomorphismGamma, EvalIsomorphism
from genericOrderEmbeddingFactorization import *

def allQuadraticIdeals(D, O, gen):
    "Generate (cyclic) horizontal ideals of norm D"
    _, X = PolynomialRing(ZZ, "X").objgen()
    minpoly = X**2 - ZZ(gen.reduced_trace())*X + ZZ(gen.reduced_norm())

    idealgens = []
    facD = factor(D)
    for l, e in facD:
        R, X = PolynomialRing(GF(l), "X").objgen()
        lambdas = R(minpoly).roots()
        gen_ls = []
        for lam, _ in lambdas:
            gen_l = gen - ZZ(lam)
            while gen_l/l in O:
                gen_l /= l
            gen_ls.append((gen_l, l))
        for _ in range(e):
            idealgens.append(gen_ls)

    for gens_set in itertools.product(*idealgens):
        Oi = O
        Ii = Oi*1
        for gens in gens_set:
            J = Oi*gens[0] + Oi*gens[1]
            Ii = Ii * J
            Oi = J.right_order()

        assert Ii.norm() == D
        yield Ii

def ConnectingIdealWithNorm(d, O1, O2):
    """
    Given an integer d and two orders O1, O2,
    find a left O1-ideal of norm d with right order isomorphic to O2
    """
    _, gamma_1, _, _ = ReducedBasis(O2)
    for omega in GenericOrderEmbeddingFactorizationAll(O1, ZZ((d*gamma_1).reduced_trace()), ZZ((d*gamma_1).reduced_norm())):
        d_m = gcd(QuaternionOrderBasis(omega, O1))
        d_mm = gcd(d_m, d) #Take care! Possible to go "above" the right level too!
        assert omega/d_m in O1 #
        I_1 = O1*(omega/d_m) + O1*(d/d_mm)
        O_crater = I_1.right_order()
        assert omega/d in O_crater
        for I_2 in allQuadraticIdeals(d_mm, O_crater, omega/d):
            if isIsomorphic(I_2.right_order(), O2):
                return I_1*I_2
    return None

def OptimalPath(ell, O1):
    #Looks for an optimal path in the ell-isogeny graph
    #from O1 to O, where O is the endomorphism ring of j-invariant 0

    B = O1.quaternion_algebra()
    i, j, k = B.gens()
    p = -ZZ(j**2)
    assert p % 12 == 11

    # Need to find O in the right quaternion algebra
    Bm = QuaternionAlgebra(-3, -p)
    ii, jj, kk = Bm.gens()
    O = Bm.quaternion_order([Bm(1), (1+ii)/2, (jj+kk)/2, (ii + kk)/3])
    gamma, _ = IsomorphismGamma(Bm, B)
    O = B.quaternion_order([EvalIsomorphism(b, B, gamma) for b in O.basis()])

    top = ceil(log(p, 2))+3
    bot = 1

    #Binary search for the minimum number of steps
    while top > bot:
        mid = (top + bot)//2
        steps = mid
        print(f"testing: {ell}^{steps}")
        alpha = None
        I = ConnectingIdealWithNorm(ell**steps, O1, O)
        if not I:
            bot = mid + 1
        else:
            I_shortest = I
            top = mid

    assert I_shortest.left_order() == O1
    assert isIsomorphic(I_shortest.right_order(), O)
    return I

