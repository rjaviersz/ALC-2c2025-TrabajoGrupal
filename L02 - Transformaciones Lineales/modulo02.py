# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 23:38:44 2025

@author: rjavi
"""

import numpy as np

# =============================================================================
# Recibe un angulo theta y retorna una matriz de 2x2
# que rota un vector dado en un angulo theta
# =============================================================================
def rota(theta) :
    lista = [np.cos(theta), -np.sin(theta),
             np.sin(theta), np.cos(theta)]
    A = np.array(lista).reshape(2,2)
    return A

# Usando coordenadas polares para un v = (x,y) cualquiera, tengo:
# x = r.cos(tita0) , y = r.sen(tita0)  *(1)

# Luego una rotacion de un angulo tita me daria un w = (x1, y1), entonces:
# x1 = r.cos(tita0 + tita) , y1 = r.sen(tita0 + tita)   *(2)

# Buscando identidades trigonometricas encuentro que:
# cos(tita0 + tita) = cos(tita0).cos(tita) - sen(tita0).sen(tita)  *(3)
# sen(tita0 + tita) = sen(tita0).cos(tita) + cos(tita0).sen(tita)  *(4)

# Uso *(3) y *(4) y los reemplazo en *(2)
# x1 = r.[ cos(tita0).cos(tita) - sen(tita0).sen(tita) ]  *(5)
# y1 = r.[ sen(tita0).cos(tita) + cos(tita0).sen(tita) ]  *(6)

# Ahora reemplazo cos(tita0) y sen(tita0) de *(5) y *(6) ,usando *(1) 
# en las ecuaciones *(5) y *(6)
# x1 = r.[ (x/r .cos(tita)) - (y/r .sen(tita)) ]
# y1 = r.[ (x/r .cos(tita)) + (y/r .sen(tita)) ]

# Me queda finalmente que el vector rotado w = (x1, y1) es igual a:
# x1 = x.cos(tita) - y.sen(tita)
# y1 = x.cos(tita) + r.sen(tita)
   

# =============================================================================
# Recibe una tira de números s y retorna una matriz cuadrada de
# n x n, donde n es el tamaño de s.
# La matriz escala la componente i de un vector de Rn en un factor s[i]
# =============================================================================
def escala(s):
    n = len(s)
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = s[i]
    return A       
# A es una matriz diagonal, de esta manera puedo escalar cada componente v[i]
# de un vector v , en un factor s[i]
    

# =============================================================================
# Recibe un ángulo theta y una tira de números s,
# y retorna una matriz de 2x2 que rota el vector en un ángulo theta
# y luego lo escala en un factor s
# =============================================================================   
def rota_y_escala(theta, s):
    A = rota(theta)
    B = escala(s)
    C = B @ A    # B.(A.x) = v  ===> (B.A).x = v  (v vector rotado y escalado)
    return C     # lo pienso como una composicion de matrices (entre B y A)

    
# =============================================================================
# Recibe un ángulo theta, una tira de números s (en R2), y un vector b en R2.
# Retorna una matriz de 3x3 que rota el vector en un ángulo theta,
# luego lo escala en un factor s y por último lo mueve en un valor fijo b
# =============================================================================
def afin(theta, s, b):
    C = rota_y_escala(theta, s)
    A = np.zeros((3,3))
    A[:2, :2] = C[::]  # Usé slicing visto en el Colab del labo00
    # con A[:2, :2] ,selecciono las primeras 2 filas, y las primeras 2 columnas
    for i in range(2):
        A[i,2] = b[i]
    A[2,2] = 1
    return A

    
# =============================================================================
# Recibe un vector v (en R2), un ángulo theta,
# una tira de números s (en R2), y un vector b en R2.
# Retorna el vector w resultante de aplicar la transformación afín a v
# =============================================================================
def trans_afin(v, theta, s, b):
    A = afin(theta, s, b)
    n, m = A.shape
    v3 = np.append(v,1)
    w = np.zeros(n-1) 
    for fila in range(n-1):
        suma = 0
        for columna in range(m):
            suma = suma + A[fila,columna]*v3[columna]
        w[fila] = suma
    return w


# =============================================================================
# Tests utilizando la funcion assert:
# =============================================================================
    
#Tests para rota
assert(np.allclose(rota(0), np.eye(2)))
assert(np.allclose(rota(np.pi/2), np.array([[0, -1], [1, 0]])))
assert(np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]])))

#Tests para escala
assert(np.allclose(escala([2,3]), np.array([[2, 0], [0, 3]])))
assert(np.allclose(escala([1,1,1]), np.eye(3)))
assert(
    np.allclose(escala([0.5,0.25]), np.array([[0.5, 0], [0, 0.25]]))
)

#Tests para rota y escala
assert(
    np.allclose(rota_y_escala(0, [2,3]), np.array([[2, 0], [0, 3]]))
)
assert(np.allclose(
    rota_y_escala(np.pi/2, [1,1]), np.array([[0, -1], [1, 0]])
))
assert(np.allclose(
    rota_y_escala(np.pi, [2,2]), np.array([[-2, 0], [0, -2]])
))

#Tests para afin
assert(np.allclose(
    afin(0, [1,1], [1,2]),
    np.array([[1, 0, 1],
              [0, 1, 2],
              [0, 0, 1]])
))

assert(np.allclose(afin(np.pi/2, [1,1], [0,0]),
    np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]])
))

assert(np.allclose(afin(0, [2,3], [1,1]),
    np.array([[2, 0, 1],
              [0, 3, 1],
              [0, 0, 1]])
))

#Tests para trans afin
assert(np.allclose(
    trans_afin(np.array([1,0]), np.pi/2, [1,1], [0,0]),
    np.array([0,1])
))
assert(np.allclose(
    trans_afin(np.array([1,1]), 0, [2,3], [0,0]),
    np.array([2,3])
))
assert(np.allclose(
    trans_afin(np.array([1,0]), np.pi/2, [3,2], [4,5]),
    np.array([4,7])
))

    



