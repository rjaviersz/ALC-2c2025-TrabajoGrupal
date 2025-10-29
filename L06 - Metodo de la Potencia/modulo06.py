# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 21:01:10 2025

@author: rjavi
"""

import numpy as np

def metpot2k(A, tol=1e-15, K=1000):
    """
    A: una matriz de n x n
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector.
    K: el número máximo de iteraciones a realizarse.
    Retorna: vector v, autovalor lambda y número de iteraciones realizadas k.
    """
    n = A.shape[0]
    v = np.random.rand(n) #vector aleatorio de n elementos
    vv = calcularAx(A, calcularAx(A, v)) 
    e = filaxColumna(vv, v)   # es un prod interno
    k_iter = 0
    while ( abs(e-1) > tol and k_iter < K ):
        v = vv
        if norma(v, 2) > 0 :
            v = v / norma(v, 2)   # normalizo
        else :
            v = 0
        vv = calcularAx(A, calcularAx(A, v))
        if norma(vv, 2) > 0 :
            vv = vv / norma(vv, 2)  # normnalizo
        else :
            vv = 0       
        e = filaxColumna(vv, v) # la idea es que en algun momento 
        # esto de 1 o muy cercano a 1. 
        # Esto quiere decir que v y vv son el mismo vector
        # v es el vector de la iteracion anterior y vv el nuevo 
        # si son el mismo o casi identicos, finalizo el ciclo y ese es el avec
        
        k_iter = k_iter+1
    l = filaxColumna(vv,(calcularAx(A, vv))) # es el autovalor de este avec
    e = e-1
    return vv, l, k_iter

    

def diagRH(A, tol=1e-15, K=1000):
    """
    A: una matriz simétrica de n x n
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector.
    K: el número máximo de iteraciones a realizarse.
    Retorna: matriz de autovectores S y matriz de autovalores D, tal que A = S D S.T.
    Si la matriz A no es simétrica, debe retornar None.
    """
    n = A.shape[0]
    autovec, autoval, _ = metpot2k(A, tol, K)
    H_v1 = matriz_Householder(autovec)
    
    if n == 2:
        S = H_v1
        D = multiplicar_matrices(H_v1, multiplicar_matrices(A, traspuesta(H_v1)))
        return S, D
    else:
        B = multiplicar_matrices(H_v1, multiplicar_matrices(A, traspuesta(H_v1)))
        A2 = B[1:n,1:n]
        S2, D2 = diagRH(A2, tol, K)
        D = np.zeros((n,n))
        D[0,0] = autoval
        for i in range(1,n):
            D[i,i] = D2[i-1,i-1]
        S3 = np.zeros((n,n)) # S3 es la matriz con 1 en el (0,0) con ceros en el resto de la fila 1 y columna 1 y S2 en S3[1:n,1:n]
        S3[0,0] = 1
        for i in range(1,n):
            for j in range(1,n):
                S3[i,j] = S2[i-1,j-1]
        S = multiplicar_matrices(H_v1, S3)
        
        return S, D

# Funciones auxiliares
def calcularAx(A, x):
    n, m = A.shape
    b = np.zeros(n) 
    for fila in range(n):
        suma = 0
        for columna in range(m):
            suma = suma + A[fila,columna]*x[columna]
        b[fila] = suma
    return b

def filaxColumna(fila, columna):   # es lo mismo que hacer producto interno
    n = fila.size
    suma = 0 
    for i in range(n) :
        suma = suma + (fila[i]*columna[i])
    return suma

def traspuesta(A): 
    n, m = A.shape
    At = np.zeros((m, n))  # matriz de retorno donde ire cambiando valores
    for columna in range(m):
        for fila in range(n):
            At[columna, fila] = A[fila, columna]
    return At

def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2. (c).
    """
    lista = []
    if p == 1 :
        A = A.T
        for fila in A:
            lista.append(norma(fila,1))
        return max(lista) 
    elif p=="inf" :
        for fila in A:
            lista.append(norma(fila,1))
        return max(lista) 
    else :
        return    
    
def norma(x,p):
    """
    Calcula la norma p del vector x sin np.linalg.norm
    """
    x = np.array(x)
    if p == 1:
        for i in range(len(x)):
            x[i] = abs(x[i])
        return sum(x)
    elif p == 2:
        for i in range(len(x)):
            x[i] = x[i]**2
        return np.sqrt(sum(x))
    elif p == "inf":
        for i in range(len(x)):
            x[i] = abs(x[i])
        return max(x)
    else:
        raise ValueError("p debe ser 1, 2 o np.inf")
    
def matriz_Householder(autovec):
    """
    Dado un autovector, devuelve la matriz de Householder asociada
    """
    n = autovec.shape[0]
    I = np.eye(n)
    e1 = np.zeros(n)
    e1[0] = 1
    vec_HH = e1 - autovec
    denominador = filaxColumna(vec_HH, vec_HH) # vec_HH·vec_HH = ||vec_HH||^2
    M = np.zeros((n,n))
    
    # Construcción de la matriz de Householder
    for i in range(n):
        for j in range(n):
            M[i, j] = I[i, j] - 2 * vec_HH[i] * vec_HH[j] / denominador
    return M

def multiplicar_matrices(A,B):
    """
    Dada una matriz A y B las multiplica
    """
    n, m = A.shape
    m2, p = B.shape
    
    AB = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            suma = 0
            for k in range(m):
                suma += A[i,k] * B[k,j]
            AB[i,j] = suma
    
    return AB


#### TESTEOS
# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95



# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95



# =============================================================================
#  Esto de abajo no va
# =============================================================================
# #### TESTEOS
# # Tests diagRH

# def rand_matriz_simetrica(n):
#     A = np.random.random((n, n))
#     for i in range(n):
#         for j in range(n):
#             A[i,j] = A[j,i]
#     return A

# TAMANIO = 5
# # c) i
# exitos = 0
# for i in range(100):
#     A = rand_matriz_simetrica(TAMANIO) #podemos ir cambiando el tamaño
#     S,D = diagRH(A,tol=1e-15,K=1e5)
#     SDS_t = multiplicar_matrices(S, multiplicar_matrices(D, traspuesta(S)))
#     e = normaExacta(SDS_t - A, p='inf')
#     if e < 1e-5:
#         exitos += 1
# assert exitos >= 95
    
# #c) ii
# suma = 0
# for i in range(100):
#     A = rand_matriz_simetrica(TAMANIO)
#     autovals = np.linalg.eigvalsh(A)
#     S,D = diagRH(A,tol=1e-15,K=1e5)
#     for i in range(TAMANIO):
#         suma += abs(D[i,i]) - abs(autovals[i])
# assert suma < 1e-15

# # d)
# N = 10
# tols = [1e-5, 1e-10, 1e-15]
# errores = []
# for t in tols:
#     for i in range(10):
#         A = rand_matriz_simetrica(N)
#         S,D = diagRH(A,tol=t,K=1e5)
#         norma_inf = normaExacta(multiplicar_matrices(S, multiplicar_matrices(D, traspuesta(S))) - A, p='inf')
#         errores.append((t, norma_inf))
# # print(errores)  # Descomentar para ver los errores obtenidos

