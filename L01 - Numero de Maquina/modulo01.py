# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 21:45:02 2025

@author: rjavi
"""
import numpy as np

# =============================================================================
# Recibe dos numeros x e y, y calcula el error de aproximar 
# x usando y en float64
# =============================================================================
def error(x,y):
    x = np.float64(x)
    y = np.float64(y)
    return abs(y-x)
    
# =============================================================================
# Recibe dos numeros x e y, y calcula el error relativo de aproximar 
# x usando y en float64
# =============================================================================
def error_relativo(x,y):
    return error(x,y) / abs(x)

# =============================================================================
# Devuelve True si ambas matrices son iguales y False en otro caso.
# Considerar que las matrices pueden tener distintas dimensiones,
# ademas de distintos valores.
# =============================================================================

def matricesIguales(A,B):
    eps = np.float64( 0.5*(np.float64(10**(-15))) ) #es el epsilon o tolerancia
    # este epsilon lo saqué del material del labo. pagina14
    if A.shape != B.shape:
        return False
    else:
        for fila in range(A.shape[0]):
            for columna in range(A.shape[1]):
                aij = np.float64(A[fila,columna])
                bij = np.float64(B[fila,columna])
                if (np.float64(aij - bij) > eps):
                    return False
        return True

# =============================================================================
# Se aportan una serie de tests utilizando la funcion assert.
# =============================================================================
def sonIguales(x,y,atol=1e-08):
    return np.allclose(error(x,y),0,atol=atol)
 
assert(not sonIguales(1,1.1))
assert(sonIguales(1,1 + np.finfo('float64').eps))
assert(not sonIguales(1,1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))

assert(np.allclose(error_relativo(1,1.1),0.1))
assert(np.allclose(error_relativo(2,1),0.5))
assert(np.allclose(error_relativo(-1,-1),0))
assert(np.allclose(error_relativo(1,-1),2))

assert(matricesIguales(np.diag([1,1]),np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

