import numpy as np

######################################################################################
##                                                                                  ##
##  Upwind operators of order 3, 5, and 7.                                          ##   
##                                                                                  ##
##  Author: Gustav Eriksson                                                         ##
##  Date:   2022-08-31                                                              ##
##                                                                                  ##
##  Based on Matlab code written by Ken Mattsson.                                   ##
##                                                                                  ##
##  The code has been tested on the following versions:                             ##
##  - Python     3.9.2                                                              ##
##  - Numpy      1.19.5                                                             ##
##                                                                                  ##
######################################################################################


# Upwind 1D third order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# from upwind import sbp_upwind_3rd
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = sbp_upwind_3rd(m,h,order)
def sbp_upwind_3rd(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:4,0:4] = np.diag(np.array([0.4347899357e10/0.12695947216e11, 0.12032349023e11/0.9521960412e10, 0.32831414215e11/0.38087841648e11, 0.6550489565e10/0.6347973608e10]))
    H[-4:,-4:] = np.fliplr(np.flipud(H[0:4,0:4]))
    H = h*H

    HI = np.linalg.inv(H)

    Qp = -1/3*np.diag(np.ones(m-1),-1) - 1/2*np.diag(np.ones(m),0) + 1*np.diag(np.ones(m-1),1) - 1/6*np.diag(np.ones(m-2),2);

    Qu = np.array([
        [-0.847e3/0.37560e5, 0.79604458492699e14/0.119214944358240e15, -0.1643521867663e13/0.14901868044780e14, -0.4160444549287e13/0.119214944358240e15],
        [-0.22671019561497e14/0.39738314786080e14, -0.6023e4/0.37560e5, 0.91628011326497e14/0.119214944358240e15, -0.749671686919e12/0.19869157393040e14],
        [0.63495586071e11/0.1241822337065e13, -0.16644840223051e14/0.39738314786080e14, -0.4311e4/0.12520e5, 0.104757273135509e15/0.119214944358240e15],
        [0.4998377065543e13/0.119214944358240e15, -0.5276507651527e13/0.59607472179120e14, -0.12476888349687e14/0.39738314786080e14, -0.5919e4/0.12520e5]])

    Qp[:4,:4] = Qu
    Qp[-4:,-4:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI@(Qp - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))
    Dm = HI@(Qm - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))

    return H,HI,Dp,Dm,e_l,e_r

# Upwind 1D fifth order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# from upwind import sbp_upwind_5th
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = sbp_upwind_5th(m,h,order)
def sbp_upwind_5th(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:4,0:4] = np.diag(np.array([0.251e3/0.720e3,0.299e3/0.240e3,0.211e3/0.240e3,0.739e3/0.720e3]))
    H[-4:,-4:] = np.fliplr(np.flipud(H[0:4,0:4]))
    H = h*H

    HI = np.linalg.inv(H)

    Qp = 1/20*np.diag(np.ones(m-2),-2) - 1/2*np.diag(np.ones(m-1),-1) - 1/3*np.diag(np.ones(m),0) + np.diag(np.ones(m-1),1) - 1/4*np.diag(np.ones(m-2),2) + 1/30*np.diag(np.ones(m-3),3)
    
    Qu = np.array([
        [-0.1e1/0.120e3, 0.941e3/0.1440e4, -0.47e2/0.360e3, -0.7e1/0.480e3],
        [-0.869e3/0.1440e4, -0.11e2/0.120e3, 0.25e2/0.32e2, -0.43e2/0.360e3],
        [0.29e2/0.360e3, -0.17e2/0.32e2, -0.29e2/0.120e3, 0.1309e4/0.1440e4],
        [0.1e1/0.32e2, -0.11e2/0.360e3, -0.661e3/0.1440e4, -0.13e2/0.40e2]])

    Qp[:4,:4] = Qu
    Qp[-4:,-4:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI@(Qp - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))
    Dm = HI@(Qm - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))

    return H,HI,Dp,Dm,e_l,e_r

# Upwind 1D seventh order accurate finite difference SBP operators.
# Input:
#   m - number of grid points (integer)
#   h - step size (float)
# 
# Output:
#   H - inner product matrix
#   HI - inverse of H
#   Dp - "positive" difference operator
#   Dm - "negative" difference operator
#   e_l,e_r - vectors to extract the boundary grid points
# 
# Use as follows:
# 
# from upwind import sbp_upwind_7th
# H,HI,D1,D2,e_l,e_r,d1_l,d1_r = sbp_upwind_7th(m,h,order)
def sbp_upwind_7th(m,h):
    e_l = np.zeros(m)
    e_l[0] = 1

    e_r = np.zeros(m)
    e_r[-1] = 1

    H = np.diag(np.ones(m))
    H[0:6,0:6] = np.diag(np.array([0.19087e5/0.60480e5,0.84199e5/0.60480e5,0.18869e5/0.30240e5,0.37621e5/0.30240e5,0.55031e5/0.60480e5,0.61343e5/0.60480e5]))
    H[-6:,-6:] = np.fliplr(np.flipud(H[0:6,0:6]))
    H = h*H

    HI = np.linalg.inv(H)

    Qp = -1/105*np.diag(np.ones(m-3),-3) + 1/10*np.diag(np.ones(m-2),-2) - 3/5*np.diag(np.ones(m-1),-1) - 1/4*np.diag(np.ones(m),0) + np.diag(np.ones(m-1),1) - 3/10*np.diag(np.ones(m-2),2) + 1/15*np.diag(np.ones(m-3),3) - 1/140*np.diag(np.ones(m-4),4);
    
    Qu = np.array([
        [-0.265e3/0.300272e6, 0.1587945773e10/0.2432203200e10, -0.1926361e7/0.25737600e8, -0.84398989e8/0.810734400e9, 0.48781961e8/0.4864406400e10, 0.3429119e7/0.202683600e9],
        [-0.1570125773e10/0.2432203200e10, -0.26517e5/0.1501360e7, 0.240029831e9/0.486440640e9, 0.202934303e9/0.972881280e9, 0.118207e6/0.13512240e8, -0.231357719e9/0.4864406400e10],
        [0.1626361e7/0.25737600e8, -0.206937767e9/0.486440640e9, -0.61067e5/0.750680e6, 0.49602727e8/0.81073440e8, -0.43783933e8/0.194576256e9, 0.51815011e8/0.810734400e9],
        [0.91418989e8/0.810734400e9, -0.53314099e8/0.194576256e9, -0.33094279e8/0.81073440e8, -0.18269e5/0.107240e6, 0.440626231e9/0.486440640e9, -0.365711063e9/0.1621468800e10],
        [-0.62551961e8/0.4864406400e10, 0.799e3/0.35280e5, 0.82588241e8/0.972881280e9, -0.279245719e9/0.486440640e9, -0.346583e6/0.1501360e7, 0.2312302333e10/0.2432203200e10],
        [-0.3375119e7/0.202683600e9, 0.202087559e9/0.4864406400e10, -0.11297731e8/0.810734400e9, 0.61008503e8/0.1621468800e10, -0.1360092253e10/0.2432203200e10, -0.10677e5/0.42896e5]
        ])

    Qp[:6,:6] = Qu
    Qp[-6:,-6:] = np.flipud(np.fliplr(Qu)).T

    Qm = -Qp.T

    Dp = HI@(Qp - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))
    Dm = HI@(Qm - 0.5*np.tensordot(e_l, e_l, axes=0) + 0.5*np.tensordot(e_r, e_r, axes=0))

    return H,HI,Dp,Dm,e_l,e_r