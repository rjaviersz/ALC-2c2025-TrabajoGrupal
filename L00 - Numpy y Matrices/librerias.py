# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""


# =============================================================================
# En un archivo librerias.py, generar la siguiente lista de funciones. Preparar
# para cada ejercicio un test que verifique que la operacion se esta 
# realizando correctamente. 
# =============================================================================

# =============================================================================
# De la libraria numpy de Python, solo estan permitidos utilizar
# las funciones que convierten listas a arrays, las que devuelven el tamaño 
# de un array y las funciones que encuentran maximos o minimos.
# =============================================================================


import numpy as np 
import numpy.linalg as lng

# =============================================================================
# Ejercicio 1. Desarrollar una funcion esCuadrada(A) que devuelva verdadero
# si la matriz A es cuadrada y Falso en caso contrario.
# =============================================================================

def esCuadrada(A):
    return len(A)== len(A[0]) # si nro Filas es igual a nro Columnas

def esCuadrada1(A):
    tamaño = A.shape
    if (tamaño[0]==tamaño[1]):
        return True
    else:
        return False
 
# === TESTS AUTOMÁTICOS ===
def test_esCuadrada():
    sizes = [(2,2), (3,3), (4,4), (3,4), (4,2), (5,6), (6,5)]  # cuadradas y no cuadradas
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        result = esCuadrada(A)  # tu implementación
        expected = n == m       # True si es cuadrada, False si no
        assert result == expected, f"Fallo en tamaño {n}x{m}"
    print("✅ 1) Todos los tests de esCuadrada pasaron correctamente.")

# Ejecutar tests
test_esCuadrada()    
 
A = np.random.randint(1,10,(4,4))
esCuadrada(A)
B = np.random.randint(0,10,(2,3))
esCuadrada(B)


# =============================================================================
# Ejercicio 2. Desarrollar una funcion triangSup(A) que devuelva la matriz U
# correspondiente a la matriz Triangular Superior de A sin su diagonal.
# =============================================================================

def triangSup(A):
    U = A.copy()
    n,m = U.shape 
    j = 1
    for fila in U:
        i = 0
        while i < j and i < m:
            fila[i] = 0
            i = i+1
        j = j + 1
    return U 

# C = np.random.randint(1,10,(3,4))
# U = triangSup(C)
# print("C :", C, "\nU : ",U)

# === TESTS AUTOMÁTICOS ===
def test_triangSup():
    sizes = [(3,3), (3,4), (4,2), (5,5), (2,5), (6,3)]  # cuadradas, n<m y n>m
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        result = triangSup(A)
        expected = np.triu(A, k=1)
        assert np.array_equal(result, expected), f"Fallo en tamaño {n}x{m}"
    print("✅ 2) Todos los tests pasaron correctamente.")

# Ejecutar tests
test_triangSup()


# =============================================================================
# Ejercicio 3. Desarrollar una funcion triangInf(A) que devuelva la matriz L
# correspondiente a la matriz Triangular Inferior de A sin su diagonal.
# =============================================================================
            
def triangInf(A):
    L = A.copy()
    n,m = L.shape 
    j = 0
    for fila in L:
        i = j
        while i < m:
            fila[i] = 0
            i = i+1 
        j = j + 1
    return L 
        
# === TESTS AUTOMÁTICOS ===
def test_triangInf():
    sizes = [(3,3), (3,4), (4,2), (5,5), (2,5), (6,3)]  # cuadradas, n<m y n>m
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        result = triangInf(A)           # tu implementación
        expected = np.tril(A, k=-1)     # referencia de NumPy
        assert np.array_equal(result, expected), f"Fallo en tamaño {n}x{m}"
    print("✅ 3) Todos los tests pasaron correctamente.")

# Ejecutar tests
test_triangInf()

# =============================================================================
#  Ejercicio 4. Desarrollar una funcioon diagonal(A) que devuelva la matriz D
#  correspondiente a la matriz diagonal de A.
# =============================================================================

def diagonal(A):
    D = A.copy()
    n, m = D.shape
    D = D - triangSup(D) - triangInf(D)
    # anula las filas extra si n > m
    if n > m:
        for i in range(m, n):
            for j in range(m):
                D[i, j] = 0
    return D

def test_diagonal():
    sizes = [(3,3), (3,4), (4,2), (5,5), (2,5), (6,3)]  # cuadradas, n<m y n>m
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        result = diagonal(A)
        # referencia robusta para cualquier tamaño
        expected = np.zeros_like(A)
        for i in range(min(n, m)):
            expected[i, i] = A[i, i]
        assert np.array_equal(result, expected), f"Fallo en tamaño {n}x{m}"
    print("✅ 4) Todos los tests pasaron correctamente.")

# Ejecutar tests
test_diagonal()

# =============================================================================
# Ejercicio 5. Desarrollar una funcion traza(A) que calcule la traza de una
# matriz cualquiera A.
# =============================================================================

def traza(A):
    minimo = min(A.shape)
    suma = 0
    for i in range(minimo):
        suma = suma + A[i,i]
    return suma 

# === TESTS AUTOMÁTICOS ===
def test_traza():
    sizes = [(3,3), (3,4), (4,2), (5,5), (2,5), (6,3)]  # cuadradas, n<m y n>m
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        result = traza(A)                 # tu implementación
        expected = np.trace(A)  # referencia de NumPy 
        assert result == expected, f"Fallo en tamaño {n}x{m}"
    print("✅ 5) Todos los tests de traza pasaron correctamente.")

# Ejecutar tests
test_traza()
    
# =============================================================================
# Ejercicio 6. Desarrollar una funcion traspuesta(A) que devuelva la matriz
# traspuesta de A.
# =============================================================================
 
def traspuesta(A): # implementacion menos costosa y mas eficiente
    n, m = A.shape
    At = np.zeros((m, n))  # matriz de retorno donde ire cambiando valores
    for columna in range(m):
        for fila in range(n):
            At[columna, fila] = A[fila, columna]
    return At
    # for fila in range(n):
    #     for columna in range(m):
    #         At[columna, fila] = A[fila, columna]

def traspuesta1(A):  # primera implementacion pero que consume mucha memoria
    At = []
    n, m = A.shape
    for columna in range(m):
        for fila in range(n):
            At.append(A[fila,columna])
    return np.array(At).reshape(m,n)
            
# === TESTS AUTOMÁTICOS ===
def test_traspuesta():
    sizes = [(3,3), (3,4), (4,2), (5,5), (2,5), (6,3)]  # cuadradas, n<m y n>m
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        result = traspuesta(A)              # tu implementación
        expected = np.transpose(A)          # referencia de NumPy
        assert np.array_equal(result, expected), f"Fallo en tamaño {n}x{m}"
    print("✅ 6) Todos los tests de traspuesta pasaron correctamente.")

# Ejecutar tests
test_traspuesta()

# =============================================================================
# Ejercicio 7. Desarrollar una funcion esSimetrica(A) que devuelve True si la
# matriz A es simetrica y False en caso contrario.
# =============================================================================
  
def esSimetrica(A):  
    # return esCuadrada(A) and (A == traspuesta(A)).all()
    return esCuadrada(A) and (np.array_equal(A, traspuesta(A)))

# === TESTS AUTOMÁTICOS ===
def test_esSimetrica():
    sizes = [(3,3), (4,4), (5,5), (3,4), (4,3), (2,5), (6,3)]
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        result = esSimetrica(A)  # tu implementación
        # referencia: simétrica solo si cuadrada y A == A.T
        if n != m:
            expected = False
        else:
            expected = np.array_equal(A, np.transpose(A))
        assert result == expected, f"Fallo en tamaño {n}x{m}"
    
    print("✅ 7) Todos los tests de esSimetrica pasaron correctamente.")

# Ejecutar tests
test_esSimetrica()

# =============================================================================
# Ejercicio 8. Desarrollar una funcion calcularAx(A,x) que recibe una matriz
# A de tamaño n×m y un vector x de largo m y devuelve un vector b de largo n
# resultado de la multiplicacion vectorial de la matriz y el vector.
# =============================================================================

def calcularAx(A, x):
    n, m = A.shape
    b = np.zeros(n) 
    for fila in range(n):
        suma = 0
        for columna in range(m):
            suma = suma + A[fila,columna]*x[columna]
        b[fila] = suma
    return b
     
# === TESTS AUTOMÁTICOS ===
def test_calcularAx():
    sizes = [(3,3), (3,4), (4,2), (5,5), (2,5), (6,3)]  # cuadradas, n<m y n>m
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        x = np.random.randint(1,10,m)  # vector de largo m
        result = calcularAx(A, x)      # tu implementación
        expected = A @ x   # multiplicación de matriz por vector con NumPy
        assert np.array_equal(result, expected), f"Fallo en tamaño {n}x{m}"
    
    print("✅ 8) Todos los tests de calcularAx pasaron correctamente.")

# Ejecutar tests
test_calcularAx()

# =============================================================================
# Ejercicio 9. Desarrollar una funcion intercambiarFilas(A, i, j), 
# que intercambie las filas i y la j de la matriz A. 
# El intercambio tiene que ser in-place.
# =============================================================================
        
def intercambiarFilas(A, i, j):
    A[[i,j], :] = A[[j,i],:] #Selecciono las filas i y j y todas las columnas
    return A   # y digo que es igual a las filas j y i y todas las columnas
       
# === TESTS AUTOMÁTICOS ===
def test_intercambiarFilas():
    sizes = [(3,3), (3,4), (4,2), (5,5), (2,5), (6,3)]  # cuadradas, n<m y n>m
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        for _ in range(3):  # probamos 3 intercambios aleatorios por matriz
            i, j = np.random.randint(0,n,2)
            A_copy = A.copy()
            intercambiarFilas(A_copy, i, j)  # tu implementación
            # referencia: intercambiar filas con copia
            expected = A.copy()
            expected[[i,j],:] = expected[[j,i],:]
            assert np.array_equal(A_copy, expected), f"Fallo en tamaño {n}x{m} intercambiando filas {i} y {j}"
    
    print("✅ 9) Todos los tests de intercambiarFilas pasaron correctamente.")

# Ejecutar tests
test_intercambiarFilas()

# =============================================================================
# Ejercicio 10. Desarrollar una funcion sumar_fila_multiplo(A, i, j, s) que
# a la fila i le sume la fila j multiplicada por un escalar s. Esta es una 
# operacion elemental clave en la eliminacion gaussiana. 
# La operacion debe ser in-place.
# =============================================================================

def sumar_fila_multiplo(A,i,j,s):
    fila_j = s*A[j]
    A[i] = A[i] + fila_j
    # A[i, :] += s * A[j, :]
    return A

# === TESTS AUTOMÁTICOS ===
def test_sumar_fila_multiplo():
    sizes = [(3,3), (3,4), (4,2), (5,5), (2,5), (6,3)]  # cuadradas, n<m y n>m
    for n, m in sizes:
        A = np.random.randint(1,10,(n,m))
        for _ in range(3):  # probamos 3 combinaciones aleatorias de filas y escalares
            i, j = np.random.randint(0,n,2)
            s = np.random.randint(-5,6)  # escalar entre -5 y 5
            A_copy = A.copy()
            sumar_fila_multiplo(A_copy, i, j, s)  # tu implementación
            
            # referencia: operación elemental
            expected = A.copy()
            expected[i, :] = expected[i, :] + s * expected[j, :]
            
            assert np.array_equal(A_copy, expected), f"Fallo en tamaño {n}x{m} sumando fila {j}*{s} a fila {i}"
    
    print("✅ 10) Todos los tests de sumar_fila_multiplo pasaron correctamente.")

# Ejecutar tests
test_sumar_fila_multiplo()

# =============================================================================
# Ejercicio 11. Desarrollar una funcion esDiagonalmenteDominante(A) que 
# devuelva True si una matriz cuadrada A es estrictamente diagonalmente 
# dominante. Esto ocurre si para cada fila, el valor absoluto del elemento en
# la diagonal es mayor que la suma de los valores absolutos de los 
# demas elementos en esa fila
# =============================================================================

def esDiagonalmenteDominante(A):
    n = A.shape[0]
    for fila in range(n):
        suma = 0
        for columna in range(n):
            if fila != columna:
               suma = suma + abs(A[fila,columna])
        if abs(A[fila,fila]) <= suma :
            return False
    return True
        
# === TESTS AUTOMÁTICOS ===
def test_esDiagonalmenteDominante():
    # casos de prueba: (matrices cuadradas y no cuadradas)
    test_cases = [
        np.array([[5,1,1],[2,6,1],[1,1,7]]),    # diagonalmente dominante
        np.array([[1,2,3],[4,5,6],[7,8,9]]),    # no dominante
        np.array([[10,0,0],[0,9,0],[0,0,8]]),   # diagonal estrictamente dominante
        np.array([[1,2],[3,4]]),                # no dominante
        np.array([[3]]),                         # 1x1, dominante
        np.random.randint(-10,10,(4,4))         # aleatoria, puede pasar o no
    ]

    expected_results = [True, False, True, False, True, None]  # None indica que es aleatoria, no sabemos a priori

    for A, expected in zip(test_cases, expected_results):
        result = esDiagonalmenteDominante(A)
        if expected is not None:
            assert result == expected, f"Fallo en matriz:\n{A}\nEsperado: {expected}, Obtenido: {result}"

    # Test con matrices cuadradas aleatorias de distintos tamaños
    sizes = [3,4,5,6]
    for n in sizes:
        A = np.random.randint(-10,10,(n,n))
        result = esDiagonalmenteDominante(A)  # tu implementación
        # validamos que devuelva True/False (no error)
        assert isinstance(result, bool), f"Fallo en tamaño {n}x{n}, no devuelve booleano"

    print("✅ 11) Todos los tests de esDiagonalmenteDominante pasaron correctamente.")

# Ejecutar tests
test_esDiagonalmenteDominante()

# =============================================================================
# Ejercicio 12. Desarrollar una funcion matrizCirculante(v) que genere una
# matriz circulante a partir de un vector. En una matriz circulante la primer 
# fila es igual al vector v, y en cada fila se encuentra una permutacion 
# cıclica de la fila anterior, moviendo los elementos un lugar hacia la derecha.
# =============================================================================

def matrizCirculante(v):
    lista = list(v)
    n = len(lista)
    A = [lista]
    for i in range(1, n):
        A.append(lista[-i:] + lista[:-i])
    return np.array(A)

# === TESTS AUTOMÁTICOS ===
def test_matrizCirculante():
    # probamos vectores de varios largos
    sizes = [2, 3, 4, 5, 6]
    for n in sizes:
        v = np.random.randint(1,10,n)
        result = matrizCirculante(v)   # tu implementación
        
        # referencia: usamos np.roll para generar cada fila
        expected = np.array([np.roll(v, k) for k in range(n)])
        
        assert np.array_equal(result, expected), f"Fallo en vector de tamaño {n}"
    
    print("✅ 12) Todos los tests de matrizCirculante pasaron correctamente.")

# Ejecutar tests
test_matrizCirculante()

# =============================================================================
# Ejercicio 13. Desarrollar una funcion matrizVandermonde(v), donde v ∈ Rn
# y se devuelve la matriz de Vandermonde V ∈ Rn×n cuya fila i-esima corresponde
# con la potencias (i − 1)-esima de los elementos de v.
# =============================================================================

def matrizVandermonde(v):
    n = v.size
    A = np.zeros((n,n))
    for fila in range(n): #ojo que empieza desde 0
        for columna in range(n): #ojo que empieza desde 0
            A[fila,columna]= (v[columna])**(fila) #como empieza en 0,no resto 1
    return A 
        

# === TESTS AUTOMÁTICOS ===
def test_matrizVandermonde():
    sizes = [2, 3, 4, 5, 6]  # probamos vectores de distintos largos
    for n in sizes:
        v = np.random.randint(1,10,n)
        result = matrizVandermonde(v)   # tu implementación

        # referencia según TU convención: fila i = v**i
        expected = np.array([v**i for i in range(n)])

        assert np.array_equal(result, expected), f"Fallo en vector de tamaño {n}\nEsperado:\n{expected}\nObtenido:\n{result}"

    print("✅ 13) Todos los tests de matrizVandermonde pasaron correctamente.")

# Ejecutar tests
test_matrizVandermonde()

# =============================================================================
# Ejercicio 16. Desarrollar una funcion matrizHilbert(n), que genera una 
# matriz de Hilbert H de n×n, y cada hij = 1/i+j+1.
# =============================================================================

def matrizHilbert(n):
    H = np.zeros((n,n))
    for fila in range(n):
        for columna in range(n):
            H[fila,columna] = 1/(fila+columna+3)
    return H 
    

# === TESTS AUTOMÁTICOS ===
def test_matrizHilbert():
    sizes = [1, 2, 3, 4, 5]  # probamos distintos tamaños
    for n in sizes:
        result = matrizHilbert(n)  # tu implementación
        expected = np.zeros((n,n), dtype=float)
        for i in range(n):
            for j in range(n):
                expected[i,j] = 1.0 / (i + j + 3)  # fórmula con indices desde 1
        assert np.allclose(result, expected), f"Fallo en tamaño {n}x{n}"
    print("✅ 16) Todos los tests de matrizHilbert pasaron correctamente.")

# Ejecutar tests
test_matrizHilbert()
    