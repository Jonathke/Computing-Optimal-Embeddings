import time
from sage.all import *
from helpers import *
from genericOrderEmbedding import *
from genericOrderEmbeddingFactorization import *
from connectingIdeal import *

def test_genericOrderEmbedding():
    print("\n\n> Testing GenericOrderEmbedding where it should be polytime")
    p = next_prime(2**150)
    
    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()
    O0 = B.quaternion_order([1, i, (i+j)/2, (1+k)/2])

    steps = 99
    omega = 2**steps*i

    I = heuristicRandomIdeal(O0, 2**steps)
    O1 = I.right_order()

    assert omega in O1

    O1, gamma = randomiseOrder(O1)
    assert gamma*omega*gamma**(-1) in O1

    t = ZZ(omega.reduced_trace())
    n = ZZ(omega.reduced_norm())

    print(f"log(p): {RR(log(p, 2))}")
    print(f"n norm: {RR(log(n, p))}")

    tstart = time.time()
    alpha = GenericOrderEmbedding(O1, t, n)
    print(f"Took {time.time() - tstart}")
    print(alpha)
    assert alpha in O1
    assert alpha.reduced_trace() == t
    assert alpha.reduced_norm() == n

    print("     > Success")

def test_genericOrderEmbeddingFactorisationHeuristic():
    print("\n\n> Testing GenericOrderEmbeddingFactorization (heuristic) with a generic order, where p is small (takes O(p^(1/3)))")
    print("These can take up to a few minutes...")
    p = next_prime(2**20) #O(p^(1/3)), takes some time
    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()
    O0 = B.quaternion_order([1, i, (i+j)/2, (1+k)/2])

    steps = 50
    omega = 2**steps*i

    I = heuristicRandomIdeal(O0, 2**steps)
    O0 = I.right_order()

    t = ZZ(0)
    n = ZZ.random_element(p**10, p**11)
    while kronecker(n, p) != 1:
        n = ZZ.random_element(p**10, p**11)

    print(f"log(p): {RR(log(p, 2))}")
    print(f"log(n,p): {RR(log(n, p))}")

    tstart = time.time()
    alpha = GenericOrderEmbeddingFactorization(O0, t, n, heuristic=True)
    print(f"Took {time.time() - tstart}")
    print(alpha)
    assert alpha in O0
    assert alpha.reduced_trace() == t
    assert alpha.reduced_norm() == n

    print("     > Success")

    print("\n\n> Testing GenericOrderEmbeddingFactorization (heuristic) with a special order")
    p = next_prime(2**150) #always be fast in this order
    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()
    O0 = B.quaternion_order([1, i, (i+j)/2, (1+k)/2]) #always be fast in this order

    t = ZZ(0)
    n = ZZ.random_element(p**3, p**4)
    while kronecker(n, p) != 1:
        n = ZZ.random_element(p**5, p**6)
    print(f"log(p): {RR(log(p, 2))}")
    print(f"log(n,p): {RR(log(n, p))}")

    tstart = time.time()
    alpha = GenericOrderEmbeddingFactorization(O0, t, n, heuristic=True)
    print(f"Took {time.time() - tstart}")
    print(alpha)
    assert alpha in O0
    assert alpha.reduced_trace() == t
    assert alpha.reduced_norm() == n

    print("     > Success")


def test_genericOrderEmbeddingFactorisation():
    print("\n\n> Testing GenericOrderEmbeddingFactorization for a generic order")
    p = next_prime(2**50)
    
    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()
    O0 = B.quaternion_order([1, i, (i+j)/2, (1+k)/2])

    steps = 33
    omega = 2**steps*i

    I = heuristicRandomIdeal(O0, 2**steps)
    O1 = I.right_order()

    assert omega in O1

    O1, gamma = randomiseOrder(O1)
    assert gamma*omega*gamma**(-1) in O1

    t = ZZ(omega.reduced_trace())
    n = ZZ(omega.reduced_norm())

    print(f"log(p): {RR(log(p, 2))}")
    print(f"n norm: {RR(log(n, p))}")

    tstart = time.time()
    alpha = GenericOrderEmbeddingFactorization(O1, t, n)
    print(f"Took {time.time() - tstart}")
    print(alpha)
    assert alpha in O1
    assert alpha.reduced_trace() == t
    assert alpha.reduced_norm() == n

    print("     > Success")

def test_optimalPath():
    print("\n\n>Finding Optimal Path")
    p = 2**55*3 - 1
    B = QuaternionAlgebra(-1, -p)

    i, j, k = B.gens()
    O1728 = B.quaternion_order([1, i, (i+j)/2, (1+k)/2])

    print(f"log(p): {RR(log(p, 2))}")
    I = OptimalPath(2, O1728)

    print(f"Ideal found, of norm {factor(I.norm())}")
    print(f"I = {I}")
    print("     > Success")

def test_connectingIdeal():
    print("\n\n> Testing ConnectingIdeal with a d = O(p^(2/3)), and one order special")
    p = next_prime(2**50)
    
    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()
    O1 = B.quaternion_order([1, i, (i+j)/2, (1+k)/2])

    d = prod(Primes()[:10])

    print(f"log(d,p): {RR(log(d, p))}")

    I = heuristicRandomIdeal(O1, d)
    O2 = I.right_order()

    _, gamma_1, _, _ = ReducedBasis(O2)
    O2, omega = randomiseOrder(O2)

    #Can do both from O2 to O1 or oposite. Run time is the same, but for different reasons!
    J = ConnectingIdealWithNorm(d, O2, O1) 

    assert J.norm() == d
    assert J.left_order() == O2
    assert isIsomorphic(J.right_order(), O1)

    print("     > Success")

def test_connectingIdealForRepresentingInteger():
    print("\n\n> Testing ConnectingIdeal with representing Integer")
    p = next_prime(2**50)
    
    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()
    O1 = B.quaternion_order([1, i, (i+j)/2, (1+k)/2])

    d = randint(0,2**70)
    I = ConnectingIdealWithNorm(d, O1, O1) 
    w = ReducedBasis(I)[0]
    assert w.reduced_norm() == d
    assert w in O1
    print("     > Success")

if __name__ == "__main__":
    test_optimalPath()
    test_connectingIdeal()
    test_connectingIdealForRepresentingInteger()
    test_genericOrderEmbedding()
    test_genericOrderEmbeddingFactorisation()
    test_genericOrderEmbeddingFactorisationHeuristic()