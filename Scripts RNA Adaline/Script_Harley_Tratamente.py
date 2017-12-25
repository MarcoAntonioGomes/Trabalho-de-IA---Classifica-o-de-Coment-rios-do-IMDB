from sklearn.datasets import *
from collections import Counter
import time
from Adaline import*


def padroniza(frequencia,frequenciaPadronizada):
    for i in range(len(frequencia)):
        frequenciaPadronizada.append((frequencia[i]-min(frequencia))/(max(frequencia)-min(frequencia)))

def padronizaClassiflicacao(frequencia, frequeciaRef):
    elementosPadronizados = []
    for i in range(len(frequencia)):
        elementosPadronizados.append((frequencia[i]-min(frequeciaRef))/(max(frequeciaRef)-min(frequeciaRef)))
    return elementosPadronizados

if __name__ == "__main__":

    print("Lendo dados de classificação...")
    categorias = ["positivo", "negativo"]



    dados = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\1º Intervalo", description = None, categories = categorias, load_content = True, shuffle = False, encoding="utf-8", decode_error = 'strict', random_state = 0)
    #dados = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\2º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)
    #dados = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\3º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)
    #dados = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\4º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)
    #dados = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\5º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)

    #dadosClassificacao = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\1º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)
    #dadosClassificacao = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\2º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)
    #dadosClassificacao = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\3º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)
    #dadosClassificacao = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\4º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)
    dadosClassificacao = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Divisão Comentarios\\5º Intervalo",description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)


    print("Ajustando os dados de classiflicação... ")
    time.sleep(100)

    comentarios = dados.data
    #print(comentarios[1])
    #print(dados.target)

    comentariosClassificacao = dadosClassificacao.data
    tamanho = len(comentarios)
    tamanhoClassificacao = len(comentariosClassificacao)
    QtdPalavras = 10000

    palavrasIgnoradas = ['herself.<br','/>If','saw','"Duchess"','sound','So','there','into','/','the','and','of','a','to','is','in','that','/><br','I','this','as','it', 'for','with','was','but','The','film','by','on','are','his','her','movie','not','you','have','be','at','from','who','an','one','he','all','she','has','or','very','their','they','so','when','some','more','-','were','what','its','my','/>The',]
    comentariosNegativos = []
    comentariosPositivos = []

    palavrasPositivasTreino = []
    palavrasNegativasTreino = []

    freqPositivasTreino = []
    freqNegativasTreino = []

    frequenciaPositivaPadronizada = []
    frequenciaNegativaPadronizada = []

    stringPositivos = ""
    stringNegativos = ""
    stringComentariosClassificacao = ""
    palavrasClassificacao = []
    freqPalavrasClassificacao = []
    elementoClassificacao = []
    taxaDeAcerto = 0
    y = []

    for i in range(int((tamanho/2)-1)):
        comentariosPositivos.append(comentarios[i])


    for i in range((int(tamanho/2)),(tamanho)):
        comentariosNegativos.append(comentarios[i])







    for i in range ((int(tamanho/2)-1)):
        stringPositivos = comentariosPositivos[i] +  stringPositivos
        stringNegativos = comentariosNegativos[i] + stringNegativos



    stringPositivos = stringPositivos.split(' ')
    stringNegativos = stringNegativos.split(' ')





    FreqNegativos = Counter(stringNegativos)
    FreqPositivos = Counter(stringPositivos)

    #print(FreqNegativos)
    #print(FreqPositivos)

    palavrasNegativas = FreqNegativos.most_common(QtdPalavras)
    palavrasPositivas = FreqPositivos.most_common(QtdPalavras)



    for i in range(QtdPalavras):
        #print(palavrasNegativas[i][0])
        if palavrasNegativas[i][0] not in palavrasIgnoradas :
            palavrasNegativasTreino.append(palavrasNegativas[i][0])
            freqNegativasTreino.append(palavrasNegativas[i][1])

        if palavrasPositivas[i][0] not in palavrasIgnoradas:
            palavrasPositivasTreino.append(palavrasPositivas[i][0])
            freqPositivasTreino.append(palavrasPositivas[i][1])


    #print( palavrasNegativasTreino)
    #print(freqNegativasTreino)
    #print( palavrasPositivasTreino)
    #print(freqPositivasTreino)

    for i in range(len(freqNegativasTreino)):

        if freqNegativasTreino[i] > freqPositivasTreino[i]:
            y.append(-1)
        else:
            y.append(1)


    padroniza(freqNegativasTreino, frequenciaNegativaPadronizada)
    padroniza(freqPositivasTreino, frequenciaPositivaPadronizada)

    #print(freqPositivasTreino)

    matriz = [frequenciaPositivaPadronizada, frequenciaNegativaPadronizada, y]


    print("Treinando a Rede Neural Adaline ... ")
    time.sleep(100)

    data = matriz
    yEsperado = y
    N =  len(freqPositivasTreino)

    setWeigths()
    trainingAndLearning(eseason,maxseason,tol,averageErrorSeasonToSeason,N,data,yEsperado)

    print("Classiflicando Comentários ... ")
    time.sleep(100)
    for i in range(tamanhoClassificacao):
        stringComentariosClassificacao = comentariosClassificacao[i]
        stringComentariosClassificacao = stringComentariosClassificacao.split(' ')
        FreqPalavrasDosComentariosClassificacao = Counter(stringComentariosClassificacao)
        FreqPalavrasDosComentariosClassificacao = FreqPalavrasDosComentariosClassificacao.most_common(len(FreqPalavrasDosComentariosClassificacao))



        for j in range(len(FreqPalavrasDosComentariosClassificacao)):
            if FreqPalavrasDosComentariosClassificacao[j][0] not in palavrasIgnoradas:

                palavrasClassificacao.append(FreqPalavrasDosComentariosClassificacao[j][0])
                freqPalavrasClassificacao.append(FreqPalavrasDosComentariosClassificacao[j][1])
                #print(palavrasClassiflicacao)
                #print(freqPalavrasClassiflicacao)

        maxElement1 = max(freqPalavrasClassificacao)
        elementoClassificacao.append(maxElement1)
        palavra1 = palavrasClassificacao[freqPalavrasClassificacao.index(maxElement1)]
        print("1° Palavra  com maior frequencia:   ", palavra1, " Frequencia: ", maxElement1)
        freqPalavrasClassificacao.remove(maxElement1)
        palavrasClassificacao.remove(palavra1)
        maxElement2 = max(freqPalavrasClassificacao)
        elementoClassificacao.append(maxElement2)
        palavra2 = palavrasClassificacao[freqPalavrasClassificacao.index(maxElement2)]
        print("2° Palavra  com maior frequencia:   ", palavra2, " Frequencia: ", maxElement2)
        elementoClassificacao = padronizaClassiflicacao(elementoClassificacao, freqPalavrasClassificacao)
        elementoClassificacao.append(bias)
        print("Elementos Padronizados: ", elementoClassificacao)

        if classification(elementoClassificacao) == -1:
            print("Comentário: Negativo ")
            if (dadosClassificacao.target[i] == 1):
                print("Classiflicou correto...")
                taxaDeAcerto += 1
            else:
                print("Classiflicou errado...")
        if classification(elementoClassificacao) == 1:
            print("Comentário: Positivo ")
            if (dadosClassificacao.target[i] == 0):
                print("Classiflicou correto...")
                taxaDeAcerto += 1
            else:
                print("Classiflicou errado...")
        elementoClassificacao = list()

    print("Taxa de Acerto: ",((taxaDeAcerto/tamanhoClassificacao)*100),"%")