import numpy as np
import math
from numpy.core.numeric import concatenate, isscalar, binary_repr, identity, asanyarray, dot
from numpy.core.numerictypes import issubdtype
from sympy import Matrix, PurePoly
from sympy.abc import x, y
import matplotlib.pyplot as plt

import warnings


def pretty_print(mat):
    for i in mat:
        print(i)


def matrix_power(M, n, mod_val):
    # Implementation shadows numpy's matrix_power, but with modulo included
    # M = asanyarray(M)
    # M = M.astype('ulonglong')
    if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("input  must be a square array")
    if not issubdtype(type(n), int):
        raise TypeError("exponent must be an integer")

    from numpy.linalg import inv

    if n == 0:
        M = M.copy()
        M[:] = identity(M.shape[0])
        return M
    elif n < 0:
        M = inv(M)
        n *= -1

    result = M % mod_val
    if n <= 3:
        for _ in range(n - 1):
            result = dot(result, M) % mod_val
        return result

    # binary decompositon to reduce the number of matrix
    # multiplications for n > 3
    beta = binary_repr(n)
    Z, q, t = M, 0, len(beta)
    while beta[t - q - 1] == '0':
        Z = dot(Z, Z) % mod_val
        q += 1
    result = Z
    for k in range(q + 1, t):
        Z = dot(Z, Z) % mod_val
        if beta[t - k - 1] == '1':
            result = dot(result, Z) % mod_val
    return result % mod_val


def py_matmult(a, b):
    zip_b = zip(*b)
    # uncomment next line if python 3 :
    zip_b = list(zip_b)

    return [[sum(int(ele_a) * int(ele_b) for ele_a, ele_b in zip(row_a, col_b))
             for col_b in zip_b] for row_a in a]



def red_mod(M, m):
    return [[int(num) % int(m) for num in row] for row in M]


def py_matrix_power(M, n, mod_val):
    if n == 0:
        M = M.copy()
        return M
    result = red_mod(M, mod_val)
    if n <= 3:
        for _ in range(n - 1):
            result = red_mod(py_matmult(result, M), mod_val)
        return result

    # binary decompositon to reduce the number of matrix
    # multiplications for n > 3
    beta = binary_repr(n)
    Z, q, t = M, 0, len(beta)
    while beta[t - q - 1] == '0':
        Z = red_mod(py_matmult(Z, Z), mod_val)
        q += 1
    result = Z
    for k in range(q + 1, t):
        Z = red_mod(py_matmult(Z, Z), mod_val)
        if beta[t - k - 1] == '1':
            result = red_mod(py_matmult(result, Z), mod_val)
    return red_mod(result, mod_val)


def py_trace(m):
    res = int()
    for n, i in enumerate(m):
        res += int(i[n])
    return res


a = [[0, 2, 1, 0],
     [2, 0, 1, 2],
     [1, 1, 0, 1],
     [0, 2, 1, 0]]


# # PRIMES_MIL contains first million primes, contains all till about 15 million
# prime_limit = 15000000 + 1
#
# with open("C:/Users/adidh/Downloads/primes1.txt", 'r') as f:
#     primes_mil = [x.strip().split() for x in f.readlines()[1:]]
#     primes_mil = [x for x in primes_mil if x != []]
#     primes_mil = [int(y) for x in primes_mil for y in x]
#     primes_mil = set(primes_mil)
#
#
# print(len(primes_mil))
#


def isprime(n):
    """Returns True if n is prime."""

    # if n in primes_mil:
    #     return True
    # return False

    if n == 2:
        return True
    if n == 3:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False

    i = 5
    w = 2

    while i * i <= n:
        if n % i == 0:
            return False

        i += w
        w = 6 - w

    return True


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


# matrix_def = np.array([
#     [0, 2, 1, 0],
#     [2, 0, 1, 2],
#     [1, 1, 0, 1],
#     [0, 2, 1, 0]]
# ).astype('ulonglong')

matrix_def = [[0, 2, 1, 0],
              [2, 0, 1, 2],
              [1, 1, 0, 1],
              [0, 2, 1, 0]]


def check_one(i, matrix=matrix_def):
    if i == 0:
        print("ZERO weirdness")
        return False
    # output_matrix = matrix_power(matrix, i, i)
    output_matrix = py_matrix_power(matrix, i, i)
    # print(i)
    # pretty_print(output_matrix)
    # print("")

    if py_trace(output_matrix) % i == py_trace(matrix) % i and not isprime(i):
        # print(output_matrix)
        print(f'Pseudoprime {i}')
        return True
    #print(f'fake {i}')
    return False


# check_one(60918)

def is_base_pseduoprime(p, a):
    if isprime(p):
        return False
    test = pow(a, p, p)
    if test == (a % p):
        return True
    return False


def main_k(skip_known=False):
    konigsberg_pseudoprimes = [1, 22, 55, 121, 242, 341, 1210, 1331, 2662, 2794, 3751, 10406, 13310, 14641, 19690, 22801, 29282, 39314, 41261, 41830, 61226, 77726, 78793, 81191, 94501, 102487, 110473, 136114, 161051, 261239, 280841, 322102, 453871, 466981, 651511, 700942, 760001, 951665, 1200905, 1213751, 1600346, 1742279, 1771561, 2817001, 3089251, 3367441, 3543122, 4295137, 4992581, 5049001, 5636785, 6733945, 7508633, 7554877, 7697921, 10386241, 10655905, 11897809, 13850386, 14154337, 18303319, 19487171, 21306157, 21459361, 23030293, 23608189, 25368497, 29618743, 33200101, 38974342, 41712209, 44136565, 46395273, 51375269, 52313521, 54449431, 54918391]

    base_2_fermat = []
    known_dodgies = [4, 9, 14, 25, 26, 33, 49, 58, 62, 91, 134, 155, 158, 161, 169, 203]

    n = 1000000000

    # if n > prime_limit:
    #     raise ValueError("above prime limit")

    # for i in range(1,n):
    #     if is_base_pseduoprime(i,3):
    #         base_2_fermat.append(i)
    #         print(i)

    for i in range(54918391+1, n+1):
        if i % 10000 == 0:
            print(i)
        if i % 100000 == 0:
            print(konigsberg_pseudoprimes)
        if skip_known:
            if len([1 for x in known_dodgies if i % x == 0]) != 0:
                continue
        if isprime(i):
            continue
        # print(i)
        if check_one(i):
            konigsberg_pseudoprimes.append(i)

    print(konigsberg_pseudoprimes)
    # print(base_2_fermat)
    # print([x for x in base_2_fermat if x in konigsberg_pseudoprimes])
    return konigsberg_pseudoprimes

def potential_speedup_k():
    konigsberg_pseudoprimes = [1, 22, 55, 121, 242, 341, 1210, 1331, 2662, 2794, 3751, 10406, 13310, 14641, 19690, 22801, 29282, 39314, 41261, 41830, 61226, 77726, 78793, 81191, 94501, 102487, 110473, 136114, 161051, 261239, 280841, 322102, 453871, 466981, 651511, 700942, 760001, 951665, 1200905, 1213751, 1600346, 1742279, 1771561, 2817001, 3089251, 3367441, 3543122, 4295137, 4992581, 5049001, 5636785, 6733945, 7508633, 7554877, 7697921, 10386241, 10655905, 11897809, 13850386, 14154337, 18303319, 19487171, 21306157, 21459361, 23030293, 23608189, 25368497, 29618743, 33200101, 38974342, 41712209, 44136565, 46395273, 51375269, 52313521, 54449431, 54918391]
    konigsberg_pseudoprimes = []

    known_dodgies = [4, 9, 14, 25, 26, 33, 49, 58, 62, 91, 134, 155, 158, 161, 169, 203]

    n = 1000000


    curr_matrix = matrix_def


    for i in range(1, n+1):
        if i % 10000 == 0:
            print(i)
            print(konigsberg_pseudoprimes)



        if isprime(i):
            pass
        else:
            if py_trace(curr_matrix) % i == 0:
                konigsberg_pseudoprimes.append(i)
        curr_matrix = py_matmult(curr_matrix, matrix_def)

    print(konigsberg_pseudoprimes)
    # print(base_2_fermat)
    # print([x for x in base_2_fermat if x in konigsberg_pseudoprimes])
    return konigsberg_pseudoprimes


def automate_main(matrix):
    power_primes = []
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    for p in primes:
        trusted = True
        power = p * p
        i = 2
        while True:
            if not check_one(power, matrix):
                trusted = False
                break
            if i == 10:
                break
            power = power * p
            i += 1
        if trusted:
            power_primes.append(p)
    return power_primes



def main(matrix = matrix_def, skip_known=False, known_dodgies = [], n=10000):
    found_pseudoprimes = []









    for i in range(1, n+1):
        if i % 10000 == 0:
            print(i)
        if i % 100000 == 0:
            print(found_pseudoprimes)
        if skip_known:
            if len([1 for x in known_dodgies if i % x == 0]) != 0:
                continue
        if isprime(i):
            continue
        # print(i)
        if check_one(i, matrix=matrix):
            found_pseudoprimes.append(i)

    # if automate:
    #     print(f'ALL PRIME POWER PSEUDOPRIMES FOUND:\n')
    #     for x in found_pseudoprimes:
    #         if len(set(prime_factors(x))) == 1:
    #             print(f'{x}: {prime_factors(x)}')
    #     print("")
    # else:

    print(f'ALL PSEUDOPRIMES FOUND:\n\n')
    for x in found_pseudoprimes:
        print(f'{x}: {prime_factors(x)}')
    print("")
    # print(base_2_fermat)
    # print([x for x in base_2_fermat if x in konigsberg_pseudoprimes])
    return found_pseudoprimes

def mini_test(m):
    a = np.array([[1, -1], [1, 1]]).astype('longdouble')
    b = np.array([[0.5, 0.5], [-0.5, 0.5]]).astype('longdouble')
    return np.matmul(a, np.matmul(m, b))


# check_one(151*151)

# print(mini_test( np.array([[14,0], [0, 4]]).astype('longdouble')))
# main(
#    [[1,1],[1,0]]
# )

# potential_speedup_k()

def epic_recurrence_mod(k, m):
    flag = False
    recs = [0, 22 % m, 24 % m]
    length = -1
    for i in range(k - 3):
        res = (recs[-2] * 11 + recs[-3] * 8) % m
        recs.append(res)
        if recs[-3:] == recs[:3] and not flag:
            length = i+1
            print(f'length={i+1}')
            flag = True


    return recs, length




def epic_recurrence(k):
    recs = [0, 22, 24]
    for i in range(k - 3):
        res = (recs[-2] * 11 + recs[-3] * 8)
        recs.append(res)
    return recs

print(epic_recurrence(100))

def generalized_recurrence(k, starting_vals, recurrence_coefficients):
    if len(starting_vals) != len(recurrence_coefficients):
        raise ValueError("Error, GENERALIZED RECURRENCE")

    recs = starting_vals.copy()
    for i in range(k-len(starting_vals)):
        res = 0
        for j in range(len(recurrence_coefficients)):
            res += recs[-j-1] * recurrence_coefficients[j]
        recs.append(res)

    return recs


def generalized_recurrence_mod_m(k, m, starting_vals, recurrence_coefficients, weird=False):
    flag = False
    if len(starting_vals) != len(recurrence_coefficients):
        raise ValueError("Error, GENERALIZED RECURRENCE")

    recs = starting_vals.copy()
    recs = [x%m for x in recs]
    length = -1
    for i in range(k-len(starting_vals)):
        res = 0
        for j in range(len(recurrence_coefficients)):
            res = (res + recs[-j-1] * recurrence_coefficients[j]) % m
        recs.append(res)
        if recs[-len(recurrence_coefficients):] == recs[:len(recurrence_coefficients)] and not flag:
            length = i+1
            print(f'length={i+1}')
            flag = True


    if weird:
        return recs, length

    return recs


#print(generalized_recurrence(10,[1,1],[1,1]))



def check_mod_m(m):
    recs = [0, 22 % m, 24 % m]
    while recs[-3:] != recs[1:4] or len(recs) <= 4:
        res = (recs[-2] * 11 + recs[-3] * 8) % m
        recs.append(res)
        # print(recs)

    # print(recs)

    x = recs[:-4] * ((math.lcm(len(recs) - 4, m)) // (len(recs) - 4))
    # print(f'full list: {x}')
    # print(f'checking multiples of m: {x[m-1::m]}')

    allowed_mods = []

    for n, a in enumerate(x[m - 1::m]):
        if a == 0:
            allowed_mods.append(((n + 1) * m) % len(x))

    allowed_mods.sort()

    print(f'Allowed residues such that n is konigsberg, and n is divisble by {m}:: mod {len(x)}: {allowed_mods}')

    if len(allowed_mods) == 0:
        return False
    return True


def no_loops(x, for_sure=4):
    pass


def check_mod_m_general(m, starting_vals, recurrence_coefficients, matrix=matrix_def):

    if len(starting_vals) != len(recurrence_coefficients):
        raise ValueError("Error, GENERALIZED RECURRENCE")

    recs = starting_vals.copy()
    recs = [x%m for x in recs]
    while recs[-3:] != recs[1:4] or len(recs) <= 4:
        if len(recs) > pow(10,5):
            print(f"ERROR, NOT PROPER CYCLIC FOR n = {m}")
            return True
        res = 0
        for j in range(len(recurrence_coefficients)):
            res = (res + recs[-j-1] * recurrence_coefficients[j]) % m
        recs.append(res)


    x = recs[:-4] * ((math.lcm(len(recs) - 4, m)) // (len(recs) - 4))
    # print(f'full list: {x}')
    # print(f'checking multiples of m: {x[m-1::m]}')

    allowed_mods = []

    for n, a in enumerate(x[m - 1::m]):
        if a == (py_trace(matrix) % m):
            allowed_mods.append(((n + 1) * m) % len(x))

    allowed_mods.sort()

    print(f'Allowed residues such that n is M-matrix pseudoprimes, and n is divisible by {m}:: mod {len(x)}: {allowed_mods}')

    if len(allowed_mods) == 0:
        return False
    return True
# check_mod_m(17*17)


def ruling_out_factors():
    banned = []

    x = 2*269

    for i in range(x,x+1):
        # print(i)
        if isprime(i):
            continue

        temp = [i % x for x in banned]
        if 0 in temp:
            continue

        if not check_mod_m(i):
            print(f'no multiples of {i}')
            banned.append(i)

    print(banned)
    return banned

def ruling_out_factors_general(starting_vals, recurrence_coefficients, matrix=matrix_def):
    banned = []
    print("\nPre-calculating disallowed modulos:\n".upper())
    for i in range(2,100):
        # print(i)
        if isprime(i):
            continue

        temp = [i % x for x in banned]
        if 0 in temp:
            continue

        if not check_mod_m_general(i, starting_vals, recurrence_coefficients, matrix):
            #print(f'no multiples of {i}')
            banned.append(i)

    #print(banned)
    return banned

def highest_power_of_11(x):
    if x == 0:
        return 'inf', 0

    ex = x % 1331

    ans = 0
    while x % 11 == 0:
        ans += 1
        x = x // 11
    return ans, ex


def checking_divisibility_by_11():
    pos_11 = []

    temp = epic_recurrence(1332)
    for n, i in enumerate(temp):
        print(f'{n + 1}: {i} ({highest_power_of_11(i)})')
        pos_11.append(highest_power_of_11(i))
    print(pos_11)

    pos_11 = [pos_11[i:i + 3] for i in range(0, len(pos_11), 3)]

    for i in pos_11:
        print(i)


# ruling_out_factors()

x = [1, 22, 55, 121, 242, 341, 1210, 1331, 2662, 2794, 3751, 10406, 13310, 14641, 19690, 22801, 29282, 39314, 41261,
     41830, 61226, 77726, 78793, 81191, 94501]


def pot_finding_more_pseudoprimes(so_far):
    for n, i in enumerate(so_far):
        print(f'{n}:{i}')
        y = py_trace(py_matrix_power(matrix_def, i, i))
        print(y)
        check_one(y, matrix_def)




def investigating_new_matrix(matrix=matrix_def):
    print("A")

    M = Matrix(matrix)
    coeffs_list = M.charpoly().all_coeffs()

    print(coeffs_list)

    starting_vals = []

    temp_matrix = matrix.copy()

    for i in range(len(matrix)):
        starting_vals.append(py_trace(temp_matrix))
        temp_matrix = py_matmult(temp_matrix, matrix)


    while coeffs_list[-1] == 0:
        coeffs_list.pop(-1)
        starting_vals.pop(-1)

    coeffs_list = coeffs_list[1:]
    coeffs_list = [-x for x in coeffs_list]



    recs = generalized_recurrence(10,starting_vals,coeffs_list)


    banned = ruling_out_factors_general(starting_vals, coeffs_list, matrix)
    print("\nThe banned multiples:\n".upper())
    print(f'{banned}\n')


    main(matrix,skip_known=True,known_dodgies=banned)

    print(f'coefficients of recurrence relation: {coeffs_list}')
    print(f'starting values {starting_vals}')
    print(f'sequence starts: {recs}')
    #[26, 437, 561, 590, 670, 1105, 1729, 5461, 6251, 6601, 8321, 8911]


def mpl_stuff(xs):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    def pseudo_pi(n):
        if n % pow(10,6) == 0:
            print(n)
        res = 0
        for i in xs:
            if i <= n:
                res += 1
        return res

    ys = [math.log(i) * math.log(math.log(i)) if math.log(i) != 0 else math.log(i) for i in range(1, xs[-1] + 1)]



    #xs = [pseudo_pi(i) / (1+math.log(pseudo_pi(i)+1,10))* (1+math.log(math.log(pseudo_pi(i)+1,10),10)) for i in range(1,xs[-1]+1)]
    xs = [pseudo_pi(i) for i in range(1, xs[-1] + 1)]


    #plt.xscale("log")
    #print(ys)
    #print(xs)


    ax.plot(ys, xs)

    plt.show()


temp_m = [
    [0,0,7],
    [1,0,7],
    [0,1,0],
]


#investigating_new_matrix(temp_m)

def semi_investigate(matrix):

    M = Matrix(matrix)
    coeffs_list = M.charpoly().all_coeffs()

    x = coeffs_list



    starting_vals = []

    temp_matrix = matrix.copy()

    for i in range(len(matrix)):
        starting_vals.append(py_trace(temp_matrix))
        temp_matrix = py_matmult(temp_matrix, matrix)

    while coeffs_list[-1] == 0:
        coeffs_list.pop(-1)
        starting_vals.pop(-1)

    coeffs_list = coeffs_list[1:]
    coeffs_list = [-x for x in coeffs_list]

    recs = generalized_recurrence(10, starting_vals, coeffs_list)


    do_it = automate_main(matrix)
    if do_it != []:
        print(f'[{x}, {do_it}],')

    # print(f'coefficients of recurrence relation: {coeffs_list}')
    # print(f'starting values {starting_vals}')
    # print(f'sequence starts: {recs}')
    # [26, 437, 561, 590, 670, 1105, 1729, 5461, 6251, 6601, 8321, 8911]


def automate_zejias_job():
    for a in range(0,11):
        for b in range(0,11):
            for c in range(11):


                mat = [
                    [0,0,0,-c],
                    [1,0,0,-b],
                    [0,1,0,-a],
                    [0,0,1,0],
                ]
                semi_investigate(mat)
    #

def convert_into_matrix(coefficients):
    #tmat = [[0 for 0 in range(len(coefficients))] for x in range(len(coefficients))]
    coefficients = coefficients[1:]
    tmat = []
    n = len(coefficients)
    for i in range(n):
        temp = []
        for j in range(n):
            if i == j+1:
                temp.append(1)
            elif j == n-1:
                temp.append(-coefficients[-i-1])
            else:
                temp.append(0)
        tmat.append(temp)
    return tmat



# x = generalized_recurrence_mod_m(1000,31,[0,22,24],[0,11,8])
# print(x)
# print(epic_recurrence_mod(5000,17))


# ruling_out_factors()
#
#
#
# automate_zejias_job()
# primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
#           109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
#           233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
#           367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491,
#           499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641,
#           643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787,
#           797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
#           947, 953, 967, 971, 977, 983, 991, 997]
#
# nums = [89,89*89]
# for p in primes:
#     nums.append(p)
#     nums.append(p*p)
#
# for p in nums:
#
#     t,length  = epic_recurrence_mod(10000000,p)
#     zeros = []
#     for n,i in enumerate(t):
#         if n > length + 1:
#             break
#         if i == 0:
#             zeros.append(n)
#
#     print(f'{p}: {-1+pow(p,3)} {len([x+1 for x in zeros])} ')
#     print(zeros)


# for i in range(10000):
#
#     print(f'11^{i}')
#     check_one(pow(11,i)*10)
#
# #automate_zejias_job()



investigating_new_matrix(convert_into_matrix([1,-26,3,3]))

def finding_promys_pseudoprime(filename):
    legit_nums = []
    promys_pseudos = []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        # print(line.split())
        line = [int(x) for x in line.split()]
        if line[1] != 3:
            legit_nums.append(line[0])

    coeffs = []
    for a in range(10):
        for b in range(10):
            coeffs.append([1,a,b])

    trial_matrices = [convert_into_matrix(x) for x in coeffs]

    print(len(legit_nums))

    for n, pot in enumerate(legit_nums):
        if n % 1000 == 0:
            print(n)
        flag = True
        for mat in trial_matrices:
            if not check_one(pot, mat):
                flag = False
                break

        if flag:

            promys_pseudos.append(pot)
            print(pot)




    print(promys_pseudos)

check_one(152046457901521, convert_into_matrix([1,1,1]))
#finding_promys_pseudoprime("carmichael_10e16_to_10e17.txt")





