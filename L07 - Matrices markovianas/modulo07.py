# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 22:31:09 2025

@author: rjavi
"""

import numpy as np

def transiciones_al_azar_continuas(n):
    """
    n: la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el
    intervalo [0,1]
    """

def transiciones_al_azar_uniforme(n, thres):
    """
    n: la cantidad de filas (columnas) de la matriz de transición.
    thres: probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. El elemento i,j es distinto de
    cero si el número generado al azar para i,j es menor o igual a thres. Todos los
    elementos de la columna j son iguales (a 1 sobre el número de elementos distintos
    de cero en la columna).
    """

def nucleo(A, tol=1e-15):
    """
    A: una matriz de m x n
    tol: la tolerancia para asumir que un vector está en el núcleo.
    Calcula el núcleo de la matriz A diagonalizando la matriz transpuesta(A) * A (* la
    multiplicación matricial), usando el método diagRH. El núcleo corresponde a los
    autovectores de autovalor con módulo <= tol.
    Retorna los autovectores en cuestión, como una matriz de n x k, con k el número de
    autovectores en el núcleo.
    """


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con índices i, lista con índices
    j, y lista con valores A_ij de la matriz A. También las dimensiones de la matriz a
    través de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir únicamente posiciones correspondientes a valores
    distintos de cero. Retorna una lista con:
    - Diccionario {(i, j): A_ij} que representa los elementos no nulos de la matriz A. Los
      elementos con módulo menor a tol deben descartarse por default.
    - Tupla (m_filas, n_columnas) que permita conocer las dimensiones de la matriz.
    """

def multiplica_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v.
    Retorna un vector w resultado de multiplicar A con v
    """


# =============================================================================
#                         # Test L06-metpot2k, Aval
# =============================================================================


#### TESTEOS
# Tests metpot2k

# S = np.vstack([
#     np.array([2,1,0])/np.sqrt(5),
#     np.array([-1,2,5])/np.sqrt(30),
#     np.array([1,-2,1])/np.sqrt(6)
#               ]).T

# # Pedimos que pase el 95% de los casos
# exitos = 0
# for i in range(100):
#     D = np.diag(np.random.random(3)+1)*100
#     A = S@D@S.T
#     v,l,_ = metpot2k(A,1e-15,1e5)
#     if np.abs(l - np.max(D))< 1e-8:
#         exitos += 1
# assert exitos > 95


# #Test con HH
# exitos = 0
# for i in range(100):
#     v = np.random.rand(9)
#     #v = np.abs(v)
#     #v = (-1) * v
#     ixv = np.argsort(-np.abs(v))
#     D = np.diag(v[ixv])
#     I = np.eye(9)
#     H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

#     A = H@D@H.T
#     v,l,_ = metpot2k(A, 1e-15, 1e5)
#     #max_eigen = abs(D[0][0])
#     if abs(l - D[0,0]) < 1e-8:         
#         exitos +=1
# assert exitos > 95



# # Tests diagRH
# D = np.diag([1,0.5,0.25])
# S = np.vstack([
#     np.array([1,-1,1])/np.sqrt(3),
#     np.array([1,1,0])/np.sqrt(2),
#     np.array([1,-1,-2])/np.sqrt(6)
#               ]).T

# A = S@D@S.T
# SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
# assert np.allclose(D,DRH)
# assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# # Pedimos que pase el 95% de los casos
# exitos = 0
# for i in range(100):
#     A = np.random.random((5,5))
#     A = 0.5*(A+A.T)
#     S,D = diagRH(A,tol=1e-15,K=1e5)
#     ARH = S@D@S.T
#     e = normaExacta(ARH-A,p='inf')
#     if e < 1e-5: 
#         exitos += 1
# assert exitos >= 95



