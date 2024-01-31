from helpers import ReducedBasis
from sage.all import *
from itertools import product

def GenericOrderEmbedding(O, t, n):
    B = O.quaternion_algebra()
    i, j, k = B.gens()
    q = -ZZ(i**2)
    p = -ZZ(j**2)

    _, beta1, beta2, beta3 = ReducedBasis(O)
    Ds = [ZZ(beta.reduced_trace()**2 - 4*beta.reduced_norm()) for beta in [beta1, beta2, beta3]]
    trace_betas = [ZZ(beta.reduced_trace()) for beta in [beta1, beta2, beta3]]

    D = t**2 - 4*n
    _, s = PolynomialRing(GF(p), "s").objgen()
    ts = [[ZZ(r) for r, e in (4*(s*s - s*ti*t) - (Di*D - (ti*t)**2)).roots()] for ti, Di in zip(trace_betas, Ds)]
    new_ts = []
    for t1, t2 in ts:
        if t1 > t2:
            new_ts.append([t2-p, t1-p, t1, t2])
        else:
            new_ts.append([t1-p, t2-p, t1, t2])
    bounds = [ceil(4*sqrt(n*betai.reduced_norm())) for betai in [beta1, beta2, beta3]]
    for i, tlist in enumerate(new_ts):
        while tlist[-2] + p < bounds[i]:
            tlist.append(tlist[-2] + p)
        tlist = [t for t in tlist if abs(t) < bounds[i]]
        tlist += [-t for t in tlist]
        tlist = list(set(tlist))

    system_eqs = []
    for beta in [B(1), beta1, beta2, beta3]:
        system_eqs.append([(gamma*beta).reduced_trace() for gamma in [1,beta1,beta2,beta3]])
        #system_eqs.append([c*scaling for c, scaling in zip(beta.coefficient_tuple(), [2, 2*(i**2), 2*(j**2), 2*(k**2)])])

    M = Matrix(QQ, system_eqs)

    for t1,t2,t3 in product(*new_ts):

        sol = M.solve_right(vector(QQ, [t, t1, t2, t3])) 
        alpha = sum([c*basis_elem for c, basis_elem in zip(sol, [1, beta1, beta2, beta3])])

        if alpha.reduced_norm() == n:
            print("Correct!")
            return alpha