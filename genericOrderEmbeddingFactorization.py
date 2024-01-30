from sage.all import *
from helpers import ReducedBasis, reduced_basis, trace_0, all_cornacchia, QuaternionOrderBasis

def O_doubleZero(O, b1, b2):
    # Compute the two dimensional sublattice of O containing elements with tr(a*b1) = 0 and tr(a*b2) = 0
    _, beta1, beta2, beta3 = ReducedBasis(O)
    system_eqs = []
    for b in [b1, b2]:
        system_eqs.append([(beta*b).reduced_trace() for beta in [1,beta1,beta2,beta3]])
    M = Matrix(QQ, system_eqs)
    basis = [sum([c*g for c,g in zip(v, [1,beta1,beta2,beta3])]) for v in M.right_kernel().basis_matrix()]
    T0 = (QQ**4).submodule([b.coefficient_tuple() for b in basis])
    M = O.free_module().intersection(T0).basis_matrix()
    return reduced_basis([sum(c*g for c,g in zip(v, O.quaternion_algebra().basis())) for v in M], O)

def _GenericOrderEmbeddingFactorizationInternal(O, t, n, heuristic=False, ans=None):
    #Algorithm 3
    B = O.quaternion_algebra()
    i, j, k = B.gens()
    q = -ZZ(i**2)
    p = -ZZ(j**2)

    delta = O.discriminant()
    beta, _, _ = trace_0(O) #Assert beta has trace 0 for simplicity
    d = ZZ(beta.reduced_norm())

    # Possible traces
    D_beta = ZZ(beta.reduced_trace()**2 - 4*beta.reduced_norm())
    D = t**2 - 4*n
    t_beta = ZZ(beta.reduced_trace())
    _, s = PolynomialRing(GF(delta), "s").objgen()
    t1s = [ZZ(r)-delta for r in (4*(s*s - s*t_beta*t) - (D_beta*D - (t_beta*t)**2)).roots(multiplicities=False)]

    #Finding gamma and the index
    basis_00 = O_doubleZero(O, B(1), beta)  
    gamma = basis_00[0]
    Z_beta_gamma = B.quaternion_order([B(1), beta, gamma, gamma*beta])
    M = Z_beta_gamma.discriminant()/delta
    print(f"M = {factor(M)}")

    # Solving for x and y
    system_eqs = []
    for b in [B(1), beta]:
        system_eqs.append([(g*b).reduced_trace() for g in [1,beta]])
    Z_beta = Matrix(QQ, system_eqs)
    
    f = BinaryQF([1, 0, d])

    bound = ceil(8*sqrt(n*beta.reduced_norm()))
    k = 0
    while not all([(t + k*delta) > bound for t in t1s]):
        for t1 in t1s:
            if ans:
                print(f"Answer: {ans.reduced_trace(), (ans*beta).reduced_trace()}")
                print(f"We're doing: {t, t1 + k*delta}")

            sol = Z_beta.solve_right(vector(QQ, [M*t, M*(t1 + k*delta)]))
            x, y = sol

            if ans:
                print(f"Answer: {QuaternionOrderBasis(M*ans, Z_beta_gamma)[:2]}")
                print(f"we're doing: {x, y}")
                true_z, true_w = QuaternionOrderBasis(M*ans, Z_beta_gamma)[2:]

            rhs = M**2*n - f(x,y)
            assert rhs % gamma.reduced_norm() == 0

            if heuristic:
                m = ZZ(rhs/(gamma.reduced_norm()*M)) #Also divide by M
                m_prime = prod([l**e for l, e in factor(m, limit=100) if l < 100])
                if not is_pseudoprime(m/(m_prime)): #check that we can factor efficiently
                    continue
                print("Found a good Cornacchia instance!")
                sols = all_cornacchia(d, m*M)
                print("Cornacchia Done!")
                if len(sols) > 0:
                    print("> M*alpha embeds!")
                for z, w in sols:
                    alpha = x + y*beta + gamma*(z + w*beta)
                    assert alpha.reduced_norm() == M**2*n
                    assert alpha.reduced_trace() == M*t
                    
                    if alpha/M in O:
                        print("Solution found!!")
                        yield alpha/M  
                print("No solution :c")
            else:
                print(f"factoring: {ZZ(rhs//gamma.reduced_norm())}")
                sols = all_cornacchia(d, ZZ(rhs/gamma.reduced_norm()))
                for z, w in sols:
                    alpha = x + y*beta + gamma*(z + w*beta)
                    assert alpha.reduced_norm() == M**2*n
                    assert alpha.reduced_trace() == M*t
                    
                    if alpha/M in O:
                        print("Solution found!!")
                        yield alpha/M
        k += 1  

# Wrappers whether to return an element or a generator object generating all elements
        
def GenericOrderEmbeddingFactorization(O, t, n, heuristic=False):
    for alpha in _GenericOrderEmbeddingFactorizationInternal(O, t, n, heuristic=heuristic):
        return alpha
    return None
    
def GenericOrderEmbeddingFactorizationAll(O, t, n):
    for alpha in _GenericOrderEmbeddingFactorizationInternal(O, t, n):
        yield alpha