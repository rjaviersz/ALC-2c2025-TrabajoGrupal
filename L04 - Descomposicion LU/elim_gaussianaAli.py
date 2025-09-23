#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def calculaLU(A):
    cant_op = 0
    m = A.shape[0]
    n = A.shape[1]
    Ac = A.copy()
    if m != n:
        print('Matriz no cuadrada')
        return None, None, cant_op
    for k in range(n - 1): #Bucle para moverse de columna en columna. Es el que mueve el pivote
        pivot = Ac[k,k] 
        if pivot == 0:
            return None, None, 0
        for i in range(k + 1, n): #Bucle para moverse de fila en fila, pero en las filas debajo del pivote
            # La división es 1 operacion
            alfa = Ac[i, k] / pivot
            cant_op += 1
            # Guardamos el multiplicador en el lugar del cero
            Ac[i, k] = alfa
            for j in range(k + 1, n): #Bucle que se mueve a lo largo de los elementos de una fila específica.
                Ac[i, j] = Ac[i, j] - alfa * Ac[k, j]# La multiplicación y la resta son 2 operaciones
                cant_op += 2
                
    ## Hasta acá, se calculó L, U y la cantidad de operaciones sobre Ac

    L = triangInfCon1s(Ac)
    U = triangSupDiag(Ac)      
          
    return L, U, cant_op

def triangSupDiag(A):
    n = A.shape[0]
    # Creo una matriz u de n*n ceros
    U = np.zeros((n, n))
    for i in range(n): #Bucle para moverse en filas
        for j in range(n): #Bucle para moverse en columnas
           if i<=j:# Si los elementos estan en la diagonal o arriba, copio el valor de A
            U[i][j]=A[i][j]
    return U

def triangInfCon1s(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    # Bucle para moverme por filas
    for i in range(n):
        # Idem pero por columnas
        for j in range(n):
            if i > j:
                # Si el elemento está debajo de la diagonal, copio el valor de A
                L[i, j] = A[i, j]
            elif i == j:
                # Si el elemento está en la diagonal, le asigno un 1
                L[i, j] = 1  
    return L

def res_tri( L , b , inferior =True ) :
    n = L.shape[0]
    y = np.zeros(n)
    
    if inferior:
        for i in range(n): # Sustitución hacia adelante
            suma= 0.0
            for j in range(i):
                coef = L[i, j]
                sol_ant = y[j]
                suma += coef * sol_ant
            
            pivote = L[i, i]
            y[i] = (b[i] - suma) / pivote
    else: # Sustitución hacia atrás
        for i in range(n - 1, -1, -1):
            
            # El valor a despejar es y[i]. Para eso, a b[i] le restamos los términos q conocemos
            suma = 0.0
            
            for j in range(i + 1, n):
                coef = L[i, j]
                sol_post = y[j] # Usamos las soluciones que ya calculamos
                suma += coef * sol_post
            
            pivote = L[i, i]
            y[i] = (b[i] - suma) / pivote
    return y

    
# Tests L04-LU
# Tests LU

L0 = np.array([[1,0,0],[0,1,0],[1,1,1]])
U0 = np.array([[10,1,0],[0,2,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(np.allclose(L,L0))
assert(np.allclose(U,U0))


L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(not np.allclose(L,L0))
assert(not np.allclose(U,U0))
assert(np.allclose(L,L0,atol=1e-3))
assert(np.allclose(U,U0,atol=1e-3))
assert(nops == 13)

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,0,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(L is None)
assert(U is None)
assert(nops == 0)
## Tests res_tri

A = np.array([[1,0,0],[1,1,0],[1,1,1]])
b = np.array([1,1,1])
assert(np.allclose(res_tri(A,b),np.array([1,0,0])))
b = np.array([0,1,0])
assert(np.allclose(res_tri(A,b),np.array([0,1,-1])))
b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b),np.array([-1,2,-2])))
b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([-1,1,-1])))

A = np.array([[3,2,1],[0,2,1],[0,0,1]])
b = np.array([3,2,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1/3,1/2,1])))

A = np.array([[1,-1,1],[0,1,-1],[0,0,1]])
b = np.array([1,0,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1,1,1])))
