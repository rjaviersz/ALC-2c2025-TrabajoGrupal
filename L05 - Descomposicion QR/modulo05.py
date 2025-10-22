import numpy as np

# BORRAR AL JUNTAR LOS MODULOS
def norma(x,p):
    """
    Calcula la norma p del vector x sin np.linalg.norm
    """
    x = np.array()
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
    
### Funciones L05-QR

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    n,m = A.shape
    
    if n != m: # Si A no es cuadrada devuelve None
        return None
    
    # Creo matriz cuadrada de ceros para Q y R
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    Qprima = np.zeros((n,n)) # Matriz Q antes de normalizar las columnas
    nops = 0 # Contador de operaciones

    Q[:,0] = A[:,0] / norma(A[:,0],2) # Primera columna de Q
    R[0,0] = norma(A[:,0],2) # Primer elemento de R
    
    
    for j in range(1,n): # Recorro columnas de A (y Q y R)
        Qprima[:,j] = A[:,j] # Copia cada columna de A a Qprima
        nops += n # Sumo 1 al numero de operaciones por copiar una columna

        for i in range(0,j): # Recorro columnas anteriores de Q
            R[i,j] = sum(Q[:,i] * Qprima[:,j]) # Calculo R[i,j]
            nops += 2*n - 1 # Operaciones del producto escalar
            Qprima[:,j] = Qprima[:,j] - R[i,j] * Q[:,i] # Actualizo Qprima
            nops += 2*n # Operaciones de la resta y multiplicacion por escalar
        
        R[j,j] = norma(Qprima[:,j],2) # Calculo R[j,j]
        nops += 2*n - 1 # Operaciones de la norma
        Q[:,j] = Qprima[:,j] / R[j,j] # Normalizo Qprima para obtener Q
        nops += n # Operaciones de la division por escalar

    if retorna_nops:
        return Q, R, nops
    return Q, R

def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """

# Tests L05-QR:

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)