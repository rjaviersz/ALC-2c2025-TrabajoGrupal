import numpy as np

def norma(x,p):
    """
    Calcula la norma p del vector x sin np.linalg.norm
    """
    if p == 1:
        for i in range(len(x)):
            x[i] = abs(x[i])
        return sum(x)
    elif p == 2:
        for i in range(len(x)):
            x[i] = x[i]**2
        return np.sqrt(sum(x))
    elif p == np.inf:
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

def normaMatMC(A,q,p,Np):
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

