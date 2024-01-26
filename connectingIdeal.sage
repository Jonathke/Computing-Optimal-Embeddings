


def OptimalPath(ell, O1):
    #Looks for an optimal path in the ell-isogeny graph
    #from O1 to O corresponding to j-invariant 0

    B = O1.quaternion_algebra()
    i, j, k = B.gens()
    p = -ZZ(j**2)
    assert p % 12 == 11

    top = ceil(log(p, 2))+3
    bot = 1

    #Binary search for the minimum number of steps
    # (1 + sqrt(-3)) / 2
    while top > bot:
        mid = (top + bot)//2
        steps = mid
        t = ell**(steps)
        n = ell**(steps*2)
        print(f"n norm: {RR(log(n, p))}")
        alpha = GenericOrderEmbeddingFactorization(O1, t, n)
        if not alpha:
            bot = mid + 1
        else:
            omega = alpha
            shortest = steps
            top = mid

    print(omega)

    assert omega in O1
    I = O1*omega + O1*(ell**shortest)
    assert omega/(ell**shortest) in I.right_order()
    return I

