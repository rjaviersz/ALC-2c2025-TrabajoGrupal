import numpy as np
"""Devuelve la norma p del vector x"""
def norma(x,p):
    if p == 1:
        suma = 0
        for elemento in x:
            suma += abs(elemento)
        return suma
    
    elif p == 2:
        suma = 0
        for elemento in x:
            suma += elemento**2
        return np.sqrt(suma)
        
    elif p == 'inf':
        # Para la norma infinita, encontramos el máximo absoluto 
        return max_abs(x)
    else: 
        # caso gral para cualquier p positivo
        suma=0
        for elemento in x:
            suma+=abs(elemento)**p
        res = suma **(1/p)
        return res
    
def max_abs(x):
    max_val=abs(x[0])
    for i in range(1,len(x),1):
        if abs(x[i])>=max_val:
            max_val=abs(x[i])
    return max_val

def normaliza(X, p):
    """
    Recibe una lista de vectores no vacíos y un escalar p.
    Devuelve una lista donde cada elemento corresponde a la normalización de los vectores de X con la norma p.
    """
    res = []
    for vector in X:
        norma_p = norma(vector, p)
        # Verificamos si la norma es cercana a cero para evitar la división por cero.
        if np.isclose(norma_p, 0):
            # Si la norma es cero (el vector es de ceros), lo devolvemos sin cambios.
            res.append(vector)
        else:
            res.append(vector / norma_p)
    return res
import numpy as np


import numpy as np

def normaMatMC2(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q,p} y el vector x en el cual se alcanza el maximo.
    Esta versión es para pasar las pruebas de laboratorio.
    """
    # A = [a11,a12,...., a1m] es de tamaño n*m
    #     [a21,a22,...., a2m]
    #     [an1,an2,...., anm]

    vectores_aleatorios = np.random.standard_normal(size=(Np, A.shape[1]))
    # Genera vectores aleatorios en una matriz de np*#columnas de A 
    #   [b11,b12,...., b1m]                       np*m
    #   [b21,b22,...., b2m]
    
    # Normaliza con la norma 'p' 
    vectores_normalizados = normaliza(vectores_aleatorios, p) # va a tener misma dimension que vectores_aleatorios
    
    max_norma = 0
    max_vector = None
    # vectores normalizados:  [c11,c12,...., c1m]  es de np*m
    #                         [c21,c22,...., c2m]
    for x in vectores_normalizados:
        # Aplica la matriz A y calcula la norma de salida 'q'
        norma_actual = norma(np.dot(A, x), q) #vector_resultante= (n*m)@(?,1) donde ?=m
        if norma_actual > max_norma:         #vector resultante va a ser de n*1
            max_norma = norma_actual
            max_vector = x

    return max_norma, max_vector

def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q,p} y el vector x en el cual se alcanza el maximo.
    Esta versión es para pasar las pruebas de laboratorio.
    """
    # La matriz A tiene un tamaño de n filas y m columnas, es decir (n x m).
    
    # Genera Np vectores aleatorios. Cada vector tiene m componentes.
    # El resultado es una matriz de (Np x m).
    vectores_aleatorios = np.random.standard_normal(size=(Np, A.shape[1]))
    
    # Normaliza cada uno de los Np vectores con la norma 'p'.
    # La dimensión de la matriz de vectores normalizados se mantiene en (Np x m).
    vectores_normalizados = normaliza(vectores_aleatorios, p)
    
    max_norma = 0
    max_vector = None
    
    for i in range(Np):
        # Selecciona el i-ésimo vector de la matriz de vectores normalizados.
        # Este vector 'y' tiene m componentes.
        y = vectores_normalizados[i]  
        
        # Aplica la matriz A al vector y.
        # La multiplicación A @ y es (n x m) @ (m x 1), que resulta en un vector (n x 1).
        vector_resultante = np.dot(A,y)
        
        # Calcula la norma 'q' del vector resultante.
        norma_actual = norma(vector_resultante, q)
        
        if norma_actual > max_norma:
            max_norma = norma_actual
            max_vector = y

    return max_norma, max_vector


# Tests L03-Normas

# Tests norma
assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

# Tests normaliza
# Tests normaliza
for x in normaliza([np.array([1]*k) for k in range(1,11)],2):
    assert(np.allclose(norma(x,2),1))
for x in normaliza([np.array([1]*k) for k in range(2,11)],1):
    assert(not np.allclose(norma(x,2),1) )
for x in normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
    assert( np.allclose(norma(x,'inf'),1) )


# Tests normaExacta
"""""
assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)),1)<=10)
assert(normaExacta(np.random.random((4,4)),'inf')<=4)
"""""
# Test normaMC

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

