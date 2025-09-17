import numpy as np

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
    
def normaliza(X, p):
    """
    Recibe, una lista de vectores no vacio, y un escalar p.
    Devuelve una lista donde cada elemento corresponde a normalizar los elementos de X con la norma p.
    """
    for i in range(len(X)):
        X[i] = X[i]/norma(X[i],p)
    return X

def normaMatMC(A, q, p, Np):  """funciona nueva version""" 
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

def normaMatMC1(A,q,p,Np):  """falla con varias pruebas"""
    """
    Devuelve la norma  ||A||{q , p} y el vector x en el cual se alcanza
    el maximo """
    vectores_aleatorios = []
    max_vector = None
    max_norma = 0
    
    for _ in range(Np):
        vector_aleatorio = np.random.rand(A.shape[1])
        vectores_aleatorios.append(vector_aleatorio)
    vectores_aleatorios = normaliza(vectores_aleatorios, p)

    for i in range(len(vectores_aleatorios)):
        Ax = A @ vectores_aleatorios[i]
        norma_Ax = norma(Ax, q)
        if norma_Ax > max_norma:
            max_norma = norma_Ax
            max_vector = vectores_aleatorios[i]
    return max_norma, max_vector

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

def condMC(A, p, Np):
    """
    Devuelve el numero de condicion de A usando la norma inducida p.
    """
    normaPdeA = normaMatMC(A,p,p,Np)[0]
    normaPdeAinv = normaMatMC(np.linalg.inv(A),p,p,Np)[0]
    nroDeCondiconDeA = normaPdeA * normaPdeAinv
    return nroDeCondiconDeA
    
def condExacta(A, p):
    """
    Que devuelva el numero de condicion de A a partir de la formula de
    la ecuacion (1) usando la norma p.
    """
    normaPdeA = normaExacta(A, p)
    normaPdeAinv = normaExacta(np.linalg.inv(A), p)
    nroDeCondicionDeA = normaPdeA * normaPdeAinv
    return nroDeCondicionDeA

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

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)),1)<=10)
assert(normaExacta(np.random.random((4,4)),'inf')<=4)

# Test normaMC

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

# Test condMC

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

# Test condExacta

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,1)
normaA_ = normaExacta(A_,1)
condA = condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,'inf')
normaA_ = normaExacta(A_,'inf')
condA = condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA))
