import itertools

def IdealGenerator(I):
    p = I.quaternion_algebra().ramified_primes()[0]
    bound = max(ceil(10*log(p,2)), 100)
    while True:
        alpha = sum([b * randint(-bound, bound) for b in I.basis()])
        if gcd(alpha.reduced_norm(), I.norm()**2) == I.norm():
            return alpha
        
def QuaternionOrderBasis(alpha, O):
    assert alpha in O
    Obasis = O.basis()
    M_O = Matrix(QQ, [b.coefficient_tuple() for b in Obasis]).transpose()
    vec_alpha = vector(alpha.coefficient_tuple())
    coeffs = M_O.inverse() * vec_alpha
    coeffs = [ZZ(c) for c in coeffs]
    #assert alpha == sum([c * beta for c, beta in zip(coeffs, Obasis)])
    return coeffs

def MakePrimitive(alpha, O):
    v_alpha = QuaternionOrderBasis(alpha, O)
    d = gcd(v_alpha)
    assert alpha/d in O
    return alpha/d
        
def Cornacchia(QF, m):
    m_prime = prod([l**e for l, e in factor(m, limit=100) if l < 100])
    if not is_pseudoprime(m/m_prime):
        return None, None, False
    sol = QF.solve_integer(m)
    if not sol:
        return None, None, False
    return sol[0], sol[1], True

def FullRepresentInteger(O0, M):
    B = O0.quaternion_algebra()
    QF = BinaryQF([1,0,1])
    i,j,k = B.gens()
    p = B.ramified_primes()[0]
    sol = None
    for _ in range(1000):
        m1 = max(floor(sqrt(round((4*M)/p, 5))), 100)
        z = randint(-m1, m1)
        m2 = floor(sqrt(round((4*M-z**2)/p, 5)))
        w = randint(-m2, m2)
        Mm = 4*M - p*QF(z,w)
        x, y, found = Cornacchia(QF, Mm)
        if not found:
            continue
        if x % 2 == w % 2 and y % 2 == z % 2 and gcd([x,y,z,w]) == 1:
            gamma = x + y*i + j*z + k*w
            gamma = MakePrimitive(gamma,O0)
            if gamma.reduced_norm() == M:
                return gamma
    return None

def heuristicRandomIdeal(O0, M):
    p = O0.quaternion_algebra().ramified_primes()[0]
    N = next_prime(randint(p, 2*p))
    alpha = FullRepresentInteger(O0, N*M)
    I = O0*alpha + O0*M
    #assert I.norm() == M
    return I

def orderIsomorphism(O, alpha):
    B = O.quaternion_algebra()
    return B.quaternion_order([alpha * g * ~alpha for g in O.basis()])

def randomiseOrder(O):
    B = O.quaternion_algebra()
    p = B.ramified_primes()[0]
    while True:
        r = sum(randrange(-p,p)*g for g in B.basis())
        if r:
            break
    return orderIsomorphism(O, r), r

def connectingIdeal(O1, O2):
    I = O1*O2
    I *= I.norm().denominator()
    return I

def gram_matrix(basis):
    M = []
    for a in basis:
        M.append([QQ(2) * a.pair(b) for b in basis])
    return Matrix(QQ, M)

def ReducedBasis(I):
    """
    Computes the Minkowski reduced basis of the
    input ideal. From LearningToSQI
    """
    def _matrix_to_gens(M, B):
        """
        Converts from a matrix to generators in the quat.
        algebra
        """
        return [sum(c * g for c, g in zip(row, B)) for row in M]

    B = I.basis()
    G = gram_matrix(I.basis())
    U = G.LLL_gram().transpose()
    reduced_basis_elements = _matrix_to_gens(U, B)
    return reduced_basis_elements


def reduced_basis(gens, O):   
    """
    More generally reduces the basis of any (not necessarily full rank) lattice
    """
    Quat = O.quaternion_algebra()
    M = matrix(list(g) for g in gens)
    big = 10^99
    scal = diagonal_matrix(ZZ, [ceil(sqrt(g.reduced_norm())*big) for g in Quat.basis()])
    M = (M * scal).LLL() * ~scal
    return [sum(c*g for c,g in zip(v, Quat.basis())) for v in M]

def _roots_mod_prime_pow(d, l, e):
    if kronecker(d, l) == -1:
        return []
    rts = Integers(l)(d).sqrt(all=True)
    rts = [ZZ(r) for r in rts]
    if l < 1000:
        for k in range(2,e+1):
            lift_rts = []
            for r in rts:
                Z_le = Integers(l**k)
                for m in range(l):
                    if Z_le(r + m*l**(k-1))**2 == Z_le(d):
                        lift_rts.append(r + m*l**(k-1))
            rts = list(set(lift_rts))
            if len(rts) == 0:
                return rts
    else:
        assert e <= 2, "e was too small"
        lift_rts = []
        if e == 2:
            for r in rts:
                x1 = (((d-r**2)/l)*pow(2*r, -1, l)) % l

                lift_r = (r + x1*l) % l**2
                assert Integers(l**2)(lift_r)**2 == Integers(l**2)(d), "lift was incorrect"
                lift_rts.append(lift_r)
            rts = lift_rts
    return rts
            

def all_roots_mod(d, M):
    fac_M = factor(M)
    
    all_rts = []
    mods = []
    for l, e in fac_M:
        _.<X> = Integers(l**e)[]
        rts = _roots_mod_prime_pow(d, l, e)
        if len(rts) == 0:
            return []
        all_rts.append(rts)
        mods.append(l**e)

    sols = []
    for crt_in in itertools.product(*all_rts): 
        sols.append(crt(list(crt_in), mods))
    
    return sols

def all_cornacchia(d, m):
    # Find all solutions to x^2 + dy^2 = m
    if m < 0:
        return []
    gfullist = [[l**k for k in range((e//2) + 1)] for l, e in factor(m/m.squarefree_part())]
    usedgs = []
    sols = []
    for glist in itertools.product(*gfullist):
        if not glist:
            g = 1
        else:
            g = prod(glist)
        if g in usedgs:
            continue
        usedgs.append(g)
        tempm = ZZ(m/(g^2))
        _.<X> = Integers(tempm)[]
        #rs = [ZZ(r) for r in (X^2 + d).roots(multiplicities=False)] #This sometimes get stuck...
        #print("Sage built in:")
        #print(rs)
        rs = all_roots_mod(-ZZ(d), tempm)
        #print("This shit")
        #print(rs)
        bound = round(tempm^(1/2),5)
        for r in rs:
            n = tempm
            while r > bound:
                n, r = r, n%r
            s = sqrt((tempm - r^2)/d)
            if s in ZZ:
                sols.append((g*r, g*s))
                sols.append((g*r, -g*s))
                sols.append((-g*r, g*s))
                sols.append((-g*r, -g*s))
                if d == 1:
                    sols.append((g*s, g*r))
                    sols.append((g*s, -g*r))
                    sols.append((-g*s, g*r))
                    sols.append((-g*s, -g*r))
    return list(set(sols))

def orderIsomorphism(O, alpha):
    B = O.quaternion_algebra()
    return B.quaternion_order([alpha * g * ~alpha for g in O.basis()])

def trace_0(I):
    beta0, beta1, beta2, beta3 = ReducedBasis(I)
    T0 = (QQ^4).submodule((QQ^4).basis()[1:])
    M = I.free_module().intersection(T0).basis_matrix()
    return reduced_basis([sum(c*g for c,g in zip(v, I.quaternion_algebra().basis())) for v in M], I)


###############################################
#                                             #
#    Jumping between quaternion algebras      #
#                                             #
###############################################

def IsomorphismGamma(B_old, B):
    r"""
    Used for computing the isomorphism between B_old and B
    See Lemma 10 [EPSV23]
    """
    if B_old == B:
        return 1
    i_old, j_old, k_old = B_old.gens()
    q_old = -ZZ(i_old**2)
    i, j, k = B.gens()
    q = -ZZ(i**2) 
    p = -ZZ(j**2)
    x, y = DiagonalQuadraticForm(QQ, [1,p]).solve(q_old/q)
    return x + j*y, (x + j_old*y)**(-1)

def EvalIsomorphism(alpha, B, gamma):
    r"""
    Given alpha \in B_old, and gamma deteremining the isomorphism from B_old to B,
    returns alpha \in B
    """
    i, j, k = B.gens()
    return sum([coeff*b for coeff, b in zip(alpha.coefficient_tuple(), [1, i*gamma, j, k*gamma])]) 


######## Connecting ########

def findIsom(O1, omega1, omega2):
    #alpha*omega1 - omega2*alpha = 0
    B = O1.quaternion_algebra()
    i,j,k = B.gens()
    a1, b1, c1, d1 = omega1.coefficient_tuple()
    a2, b2, c2, d2 = omega2.coefficient_tuple()

    System = [[a1-a2, i**2*(b1-b2), j**2*(c1 - c2), k**2*(d1 - d2)],
              [(b1*i - i*b2)/i, (i*a1 - a2*i)/i, (j*k*d1 - k*d2*j)/i, (k*c1*j - c2*j*k)/i],
              [(c1*j - c2*j)/j, (i*d1*k - d2*k*i)/j, (c1*j - c2*j)/j, (k*b1*i - b2*i*k)/j],
              [(d1*k - d2*k)/k, (i*c1*j - c2*j*i)/k, (j*b1*i - b2*i*j)/k, (k*a1 - a2*k)/k]]
    alpha = list(Matrix(QQ, System).right_kernel().basis_matrix())[0]
    assert alpha != 0

    return sum(c*g for c,g in zip(alpha, B.basis()))

def MakeCyclic(I):
    O = I.left_order()
    d = gcd([gcd(QuaternionOrderBasis(beta, O)) for beta in I.basis()])
    return I*(1/d)

def isIsomorphic(O1, O2):
    I = connectingIdeal(O1, O2)
    for alpha in ReducedBasis(I):
        if alpha.reduced_norm() == I.norm():
            return True
    return False

def basis_matrix(O):
    M_O = Matrix(QQ, [ai.coefficient_tuple() for ai in O.gens()])
    return M_O