#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR

    for k in range(n-1): # aca ire viendo si los aii menos el ultimo, son != 0
        pivot = Ac[k,k] 
        if pivot != 0:
            for i in range(k+1, m): # con esto me muevo por las filas
                alfa = Ac[i,k] / pivot
                Ac[i,k] = alfa  # voy armando L in-place
                cant_op = cant_op + 1
                for j in range(k+1,n): # con esto me muevo por las columnas
                    Ac[i,j] = Ac[i,j] - alfa*Ac[k,j]
                    cant_op = cant_op + 2
            
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
    L = triangInfCon1s(Ac)
    U = triangSupDiag(Ac)      
          
    return L, U, cant_op

def triangSupDiag(A):  # funcion auxiliar
    U = A.copy()
    n,m = U.shape 
    j = 0
    for fila in U:
        i = 0
        while i < j and i < m:
            fila[i] = 0
            i = i + 1
        j = j + 1
    return U 

def triangInfCon1s(A):
    L = A.copy()
    n,m = L.shape 
    j = 0
    for fila in L:
        i = j
        while j<m and i<m :
            fila[i] = 0
            i = i+1 
            fila[j] = 1
        j = j + 1
    return L 


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
    
