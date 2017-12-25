from sklearn.datasets import *
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB


def padroniza(frequencia,frequenciaPadronizada):
    for i in range(len(frequencia)):
        frequenciaPadronizada.append((frequencia[i]-min(frequencia))/(max(frequencia)-min(frequencia)))

def yOut(y,freqPos,freqNeg):
    for i in range(len(freqPos)):
        if freqPos[i] > freqNeg[i]:
            y.append(1)
        else:
            y.append(-1)


if __name__ == "__main__":
    categorias = ["positivo", "negativo"]
    dados = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\Sub Divisao comentarios\\1", description = None, categories = categorias, load_content = True, shuffle = False, decode_error = 'strict', random_state = 0)

    comentarios = dados.data
    vetorizador = CountVectorizer()
    comentarios_fit = vetorizador.fit_transform(comentarios)
    print(comentarios_fit)

    sacola = vetorizador.get_feature_names()
    frequenciaPositiva = []
    frequenciaNegativa = []
    y = []

    frequenciaPositivaPadronizada = []
    frequenciaNegativaPadronizada = []

    #print(sacola)




    for i in range (len(sacola)):
        soma = 0
        for j in range (249):
            soma = soma + comentarios_fit[(j, i)]
           # print("comentarios_fit[(",j,",",i,")]")
        frequenciaPositiva.append(soma)

    for i in range(len(sacola)):
        soma = 0
        for j in range(250, 499):
            soma = soma + comentarios_fit[(j, i)]
            #print("!!comentarios_fit[(", j, ",", i, ")]")
        frequenciaNegativa.append(soma)




    # print(frequenciaNegativa)
    # print(frequenciaPositiva)

    yOut(y,frequenciaPositiva,frequenciaNegativa)

    #print(y)

    padroniza(frequenciaNegativa,frequenciaNegativaPadronizada)
    padroniza(frequenciaPositiva,frequenciaPositivaPadronizada)

    #print(frequenciaNegativaPadronizada)
    #print(frequenciaPositivaPadronizada)



    matriz = [sacola, frequenciaPositivaPadronizada, frequenciaNegativaPadronizada,y]

    print(matriz)

    for i in range(4):
        for j in range(len(sacola)):
            print(matriz[i][j]," ",end = '')
        print("\n")







