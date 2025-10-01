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
    
    
    

def diagRH(A, tol=1e-15, K=1000):
    """
    A: una matriz simétrica de n x n
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimación del autovector.
    K: el número máximo de iteraciones a realizarse.
    Retorna: matriz de autovectores S y matriz de autovalores D, tal que A = S D S.T.
    Si la matriz A no es simétrica, debe retornar None.
    """
 
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


