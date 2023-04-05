import numpy as np
import scipy.linalg as splg
import scipy.sparse as spsp

######################################################################################
##                                                                                  ##
##  Periodic explicit operators of order 2, 4, 6, 8, 10, and 12, and periodic       ##
##  implicit operators.                                                             ## 
##                                                                                  ##
##  Author: Gustav Eriksson                                                         ##
##  Date:   2022-08-31                                                              ##
##                                                                                  ##
##  Based on Matlab code written by Ken Mattsson.                                   ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python     3.9.2                                                              ##
##  - Numpy      1.19.5                                                             ##
##  - Scipy      1.7.0                                                              ##
##                                                                                  ##
######################################################################################

# Central 1D finite difference explicit and periodic operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   order - order of accuracy (2,4,6,8,10 or 12)
#   use_AD - if including artificial dissipation
# 
# Output:
#   H - inner product matrix 
#   Q - skew symmetric part of D1 = inv(H)*Q
# 
# Use as follows:
# 
# from periodic import periodic_expl
# H,Q = periodic_expl(m,h,order,use_AD)
# 
def periodic_expl(m,h,order,use_AD=False):
    if order == 2:
        d = np.array([-0.5,0,0.5])
        l = 1
        r = 1
    elif order == 4:
        d = np.array([1./12,-2./3,0,2./3,-1./12])
        l = 2
        r = 2
    elif order == 6:
        d = np.array([-1./60,3./20,-3./4,0,3./4,-3./20,1./60])
        l = 3
        r = 3
    elif order == 8:
        d = np.array([1./280,-4./105,1./5,-4./5,0,4./5,-1./5,4./105,-1./280])
        l = 4
        r = 4
    elif order == 10:
        d = np.array([-1./1260,5./504,-5./84,5./21,-5./6,0,5./6,-5./21,5./84,-5./504,1./1260])
        l = 5
        r = 5
    elif order == 12:
        d = np.array([1./5544,-1./385,1./56,-5./63,15./56,-6./7,0,6./7,-15./56,5./63,-1./56,1./385,-1./5544])
        l = 6
        r = 6
    else:
        raise NotImplementedError('Order not implemented.')

    v = np.zeros(m)
    for i in range(r+1):
        v[i] = d[i+l]
    for i in range(l):
        v[m-i-1] = d[l-i-1]

    Q = spsp.csc_matrix(splg.toeplitz(np.roll(np.flip(v),1),v))
    H = spsp.csc_matrix(h*np.eye(m))

    if use_AD:
        if order == 2:
            d = np.array([1,-2,1])
            l = 1
            r = 1
            a = 0.5
        elif order == 4:
            d = -np.array([1, -4, 6, -4, 1])
            l = 2
            r = 2
            a = 1./12
        elif order == 6:
            d = np.array([1, -6, 15, -20, 15, -6, 1])
            l = 3
            r = 3
            a = 1./60
        elif order == 8:
            d = -np.array([1, -8, 28, -56, 70, -56, 28, -8, 1])
            l = 4
            r = 4
            a = 1./280
        elif order == 10:
            d = np.array([1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1])
            l = 5
            r = 5
            a = 1./1260
        elif order == 12:
            d = -np.array([1, -12, 66, -220, 495, -792, 924, -792, 495, -220, 66,-12, 1])
            l = 6
            r = 6
            a = 1./5544
        else:
            raise NotImplementedError('Order not implemented.')

        v = np.zeros(m)
        for i in range(r+1):
            v[i] = d[i+l]
        for i in range(l):
            v[m-i-1] = d[l-i-1]

        S = spsp.csc_matrix(a*splg.toeplitz(np.roll(np.flip(v),1),v))

        Q = Q - S            

    return H,Q

# Central 1D finite difference implicit and periodic operators. 
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   use_AD - if including artificial dissipation
# 
# Output:
#   H - inner product matrix 
#   Q - skew symmetric part of D1 = inv(H)*Q
# 
# Use as follows:
# 
# from periodic import periodic_imp
# H,Q = periodic_imp(m,h,use_AD)
# 
def periodic_imp(m,h,use_AD=False):

    h0 = 4203267613564094932432577824954./7049220443079284250976145948443;
    h1 = 22618790744689935699264926210401./84590645316951411011713751381316;
    h2 = -2209778222820418388602425303685./42295322658475705505856875690658;
    h3 = -1581945765./75409415044;
    h4 = 228992488./33235651987;
    h5 = 27214243./33751459947;

    q1 = 9607266784889201296177./19560081711822931675052;
    q2 = 8866705546306148289391./97800408559114658375260;
    q3 = -19659090145677941034997./293401225677343975125780;
    q4 = 127051314./37983174851;
    q5 = 389910724./128741750713;

    # Q
    l = 5
    r = 5
    d = np.array([-q5, -q4, -q3, -q2, -q1, 0, q1, q2, q3, q4, q5])

    v = np.zeros(m)
    for i in range(r+1):
        v[i] = d[i+l]
    for i in range(l):
        v[m-i-1] = d[l-i-1]

    Q = spsp.csc_matrix(splg.toeplitz(np.roll(np.flip(v),1),v))

    # H
    l = 5
    r = 5
    d = np.array([h5, h4, h3, h2, h1, h0, h1, h2, h3, h4, h5])

    v = np.zeros(m)
    for i in range(r+1):
        v[i] = d[i+l]
    for i in range(l):
        v[m-i-1] = d[l-i-1]

    H = spsp.csc_matrix(h*splg.toeplitz(np.roll(np.flip(v),1),v))

    if use_AD:
        d = -np.array([1, -12, 66, -220, 495, -792, 924, -792, 495, -220, 66,-12, 1])
        l = 6
        r = 6
        a = 1./5544

        v = np.zeros(m)
        for i in range(r+1):
            v[i] = d[i+l]
        for i in range(l):
            v[m-i-1] = d[l-i-1]

        S = spsp.csc_matrix(a*splg.toeplitz(np.roll(np.flip(v),1),v))

        Q = Q - S

    return H,Q

# Central 1D finite difference variable coefficient second derivative periodic operator.
# Constructed from the corresponding D1 operators as follows: D2(c) = D1*diag(c)*D1.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
#   order - order of accuracy (2,4,6,8,10 or 12)
# 
# Output:
#   M_fun - function computing the variable coefficient matrix M
# 
# Use as follows:
# 
# from periodic import periodic_variable_wide
# M_fun = periodic_variable_wide(m,h,order)
# 
def periodic_variable_wide(m,h,order):
    H,Q = periodic_expl(m,h,order,False)
    def M_fun(c):
        C = np.diag(c)
        M = -1/h*Q@C@Q
        return M

    return M_fun
