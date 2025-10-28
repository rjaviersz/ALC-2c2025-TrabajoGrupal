import numpy as np

# BORRAR AL JUNTAR LOS MODULOS
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

    
### Funciones L05-QR
def calcula_w(u, A): #Calculo w = u_t*A
        m = A.shape[0]
        n= A.shape[1]
        w = np.zeros(n)
        for j in range (0,n,1):
          suma=0
          for i in range(0,m,1):
            suma+=u[i]*A[i,j]
          w[j]=suma
        return w
        
def producto_exterior(v,w):
    a= v.shape[0]
    d=w.shape[0] #(tamaño de w, vector 1D) es un vector de 1*varias columnas
    matriz_res=np.zeros((a,d))
    for j in range (d):
         for i in range (a):
          matriz_res[i,j]=v[i]*w[j]
    return matriz_res #matriz del tamaño a*d

def multiplica_matrices(A, B):
    """""
    A es m x p y B es p x n.
    """
    m, p1 = A.shape
    p2, n = B.shape
    
    if p1 != p2:
        return None
    C = np.zeros((m, n))
    
    # Bucle i: filas de A (m)
    for i in range(m):
        # Bucle j: columnas de B (n)
        for j in range(n):
            suma = 0
            # Bucle k: suma de productos internos (p)
            for k in range(p1):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma
            
    return C

def calcula_M (escalar, v, w): 
        m=v.shape[0]
        n=w.shape[0] #(tamaño de w, vector 1D) es un vector de 1*varias columnas
        matriz_res=np.zeros((m,n))
        for j in range (n):
            for i in range (m):
                matriz_res[i,j]=v[i]*w[j]*escalar
        return matriz_res


def suma_vectorial(x):
    """
    Calcula la suma de todos los elementos del vector x
    """
    x = np.array(x) 
    acumulador = 0.0

    for elemento in x:
        acumulador += elemento
        
    return acumulador

def norma_2_sin_modificar(x):
    norma_sq = suma_vectorial(x**2)
    return np.sqrt(norma_sq)

def calcula_v_beta(x):
    """
    Calcula el vector de Householder (v) y el escalar beta.
    """
    
    # 1. Cálculo de alpha (con estabilidad)
    norma_x = norma_2_sin_modificar(x)
    x1 = x[0]
    
    # Aplicamos la fórmula de estabilidad: alpha = -sign(x1) * ||x||
    # Manejamos explícitamente el caso x[0] == 0 si es q no podemos usar np.linalg.norm
    if x1 == 0:
        alpha = -norma_x 
    else:
        # Esto implementa la regla de estabilidad: signo opuesto a x[0]
        alpha = -np.sign(x1) * norma_x 

    # 2. Cálculo del vector v = x - alpha * e1
    e1 = np.zeros(x.shape) 
    
    # Configuramos el primer elemento (el pivote) a 1.
    e1[0] = 1.0
    
    # v = x - alpha * e1
    v = x - alpha * e1
    
    # 3. Cálculo de beta = 2 / ||v||^2
    norma_v_sq = suma_vectorial(v**2) 
    
    if norma_v_sq < 1e-15:
        return v, 0.0

    beta = 2.0 / norma_v_sq

    return v, beta

def construye_H_k(v_sub, beta, m, k):
    """Construye la matriz H_k completa (m x m) ."""
    v_completo = np.zeros(m)
    
    # Insertar el vector de Householder v_sub en la posición correcta (k:)
    v_completo[k:] = v_sub 
    
    # M = beta * v_completo * v_completo^T usando calcula_M
    # v_completo se usa dos veces: como vector columna (v) y como vector fila (w)
    M_completa = calcula_M(beta, v_completo, v_completo) 
    
    # H_k = I - M
    I = np.eye(m)
    H_k = I - M_completa
    
    return H_k

def QR_con_HH(A, tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    #A (de n*m ), Q (de m*m) R(de m*n)--> QR=A
    A = np.copy(A) 
    m, n = A.shape 
    #Si A es una matriz Ancha
    if m < n:
        return None

    reflectores = []
    
    # 2. Bucle Principal para calcular R (k va de 0 hasta n - 1)
    #Entra al caso A una matriz alta (m>=n), como no hay más de n columnas para triangularizar. Entonces el numero max de pasos es n.
    for k in range(n-1): 
        
        # a. Definir el subvector x y calcular v_sub, beta
        x_sub = A[k:, k]
        v_sub, beta = calcula_v_beta(x_sub) 
        
        # Si beta es cero (columna ya es cero o pívot nulo), simplemente pasa al siguiente k
        if beta !=0:
        # b. Guardar el reflector para la reconstrucción de Q
            reflectores.append((v_sub, beta, k))
            
            # c. Aplicación eficiente para actualizar R (R = R - M)
            
            R_sub = A[k:, k:]
            
            # w = v^T * R_sub
            w = calcula_w(v_sub, R_sub)
            
            # M = beta * v * w
            M = calcula_M(beta, v_sub, w)
            
            # Actualizar R_sub in place: R_sub = R_sub - M
            A[k:, k:] -= M
        
    # R es la matriz A modificada. Se Aplica tolerancia final.
    R = A
    R[np.abs(R) < tol] = 0.0

    # 3. Reconstrucción de Q (Método 1: Producto Explícito Q = H0 * H1 * ...)
    Q = np.eye(m) 

    # El bucle va hacia adelante (k=0, 1, ...)
    for v_sub, beta, k in reflectores:
        
        # Construir H_k completa (m x m) usando la posición guardada k

        H_k = construye_H_k(v_sub, beta, m, k)

        # Acumulación: Q = Q @ H_k 
        Q = multiplica_matrices(Q, H_k)
    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    metodo = metodo.upper() #MAYUSCULAS
    
    if metodo == 'RH':
        # Retorna Q y R calculadas con Householder
        # La función QR_con_HH no tiene un contador de operaciones
        return QR_con_HH(A, tol)
    
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    
    else:
        # Si el método no es 'RH' ni 'GS'
        return None


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