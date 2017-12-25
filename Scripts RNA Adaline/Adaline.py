from random import  uniform
import random


eta = 0.001              # Taxa de Aprendizagem
maxseason = 3000         # Maximo de Épocas
tol = 0.05               # Tolerância ao erro
eseason = tol + 1        # Erro por época
N = 0                 # Quantidade de dados
averageErrorSeasonToSeason = list() #Erro Médio Epoca a Epoca
data = list()              # Matriz de Dados
yEsperado = list()                 # Valores esperados/
W = [] # Vetor(lista) de Pesos - W[0]  é o bias
bias = -1

def degreeBipole(u):
    if( u >= 0):
        return 1
    return -1




def calcActivationThreshold(element):
    u = 0
    for j in range(len(element)):
        u =  u + W[j]*element[j]
    return u

def calcDelta(error,element):
    delta = list()
    for k in range(len(W)):
        delta.append(error * eta * element[k])
    return  delta

def updateWeights(delta):
    for j in range(len(delta)):
        W[j] = W[j] + delta[j]

def setWeigths():
    for i in range(3):
        W.append(uniform(-0.3,0.3))

def extractElement(index, data):
    element = []
    for m in data:
      element.append(m[index])
    return element

def trainingAndLearning(eseason,maxseason,tol,averageErrorSeasonToSeason,N,data,yEsperado):
    nseason = 0  # Numero de Épocas
    index = list(range(N))
    random.shuffle(index)

    while (( nseason < maxseason) and (eseason > tol)):

        convergenceError = 0

        for i in range(N):

            currentIndex = index[i]
            element = extractElement(currentIndex,data)
            u  = calcActivationThreshold(element)
            error = yEsperado[currentIndex]-u
            delta = calcDelta(error,element)
            updateWeights(delta)
            convergenceError += (error**2)
            #print(W)

        averageErrorSeasonToSeason.append((convergenceError/N))
        eseason = averageErrorSeasonToSeason[nseason]
        nseason += 1


def classification (element):
    u = calcActivationThreshold(element)
    yhat = degreeBipole(u)
    return yhat

