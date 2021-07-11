from sympy.matrices import Matrix, eye
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.abc import z
from sympy import cancel, ratsimp
import sympy
M = Matrix([
        [z,0,0],
        [0,1,0],
        [0,0,1],
        ])
(z * eye(3) + M).inv()
M.inv()

import control as ctrl
import numpy as np

G = Matrix([
        [ (z-1.)/(z-2.), 2./(z+1.) ],
        [ z/(z+1.)   , 0 ]
        ])

tf_num = []
tf_den = []
for rowIdx in range(2):
        tf_num.append([])
        tf_den.append([])
        for colIdx in range(2):
            num, den = cancel(G[rowIdx, colIdx]).as_numer_denom()
            print('\n{}\n------\n{}\n'.format(num, den))
            if num.is_constant():
                tf_num[rowIdx].append(np.asarray([num], dtype=float))
            else:
                tf_num[rowIdx].append(np.asarray(num.as_poly().all_coeffs(), dtype=float))
            print('\n{}'.format(tf_num[rowIdx][colIdx]))
            if den.is_constant():
                tf_den[rowIdx].append(np.asarray([den], dtype=float))
            else:
                tf_den[rowIdx].append(np.asarray(den.as_poly().all_coeffs(), dtype=float))
            print('\n{}'.format(tf_den[rowIdx][colIdx]))

trFun = ctrl.tf(tf_num, tf_den, dt=.1)
sS = ctrl.ss(trFun)
eps = np.spacing(10)
sS.A[np.abs(sS.A) < eps] = 0.
sS.B[np.abs(sS.B) < eps] = 0.
sS.C[np.abs(sS.C) < eps] = 0.
sS.D[np.abs(sS.D) < eps] = 0.
A = Matrix(sS.A)
B = Matrix(sS.B)
C = Matrix(sS.C)
D = Matrix(sS.D)
Grec = C * (z * eye(*A.shape) - A).inv() * B + D

for rowIdx in range(2):
        for colIdx in range(2):
                recNum, recDen = cancel(Grec[rowIdx, colIdx]).as_numer_denom()
                print('\n{}\n------\n{}\n'.format(recNum, recDen))

recTRF = ctrl.tf(sS)
for rowIdx in range(2):
        for colIdx in range(2):
            recNum = recTRF.num[rowIdx][colIdx]
            oNum = trFun.num[rowIdx][colIdx]
            print('oNum: {}'.format(oNum))
            # np.abs(recNum - oNum)
            #
            recDen = recTRF.den[rowIdx][colIdx]
            oDen = trFun.den[rowIdx][colIdx]
            # np.abs(recDen - oDen)
            print('oDen: {}'.format(oDen))
            break
        break