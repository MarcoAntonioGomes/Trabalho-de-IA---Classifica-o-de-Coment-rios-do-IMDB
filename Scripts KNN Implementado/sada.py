from sklearn.datasets import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import os
import sys
from collections import Counter


def treino(dados):
    if(os.path.exists('treino.npz')):
        return load_sparse_csr("treino.npz")
    else:
        print("gerando treino")
        print("Digite o nome das pastas dos comentarios. Ex: Positivo, Negativo")
        print("Pasta de comentarios Positivos :")
        a = input()
        print("Pasta de comentarios Negativos :")
        b = input()

        print("Digite o diretorio do local das pastas com arquivos")
        print("Exemplo")

        print("treinando")

        print(type(dados))
        vectorizer = TfidfVectorizer()
        # tokenize and build vocab
        vec = vectorizer.fit_transform(dados.data)
        print("data", dados.data)
        save_sparse_csr("treino", vec)
        return vec


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def preparacomentarios(comentarios):
    lista_preparada = list()

    for comentario in comentarios:
        lista_preparada.append(comentario.lower())
    obj = Counter(lista_preparada)

    listafinal = list(obj)
    print(listafinal)

    return listafinal


def


def predicao(treino, teste, numero_k, vectorizer, comentario):

    # splita comentario
    comentarios = (comentario.split(" "))

    lista_preparada = preparacomentarios(comentarios)

    # lista de tupla que indica o indice de vectorizer em treino e a palavra origem
    listaBolada = list()
    print("tamanho", len(lista_preparada))
    tam = len(lista_preparada)
    c = 0
    # uma ideia é substituir esse  modelo de percorrer por um hash
    for string in lista_preparada:
        print("string =", string)
        contador = 0
        c += 1
        for j in range(len(vectorizer.get_feature_names())):
            print(c, "/", tam)
            print(len(vectorizer.get_feature_names()))
            print("featurename", vectorizer.get_feature_names()[j])
            contador += 1
            print(contador)
            if(string == vectorizer.get_feature_names()[j]):
                indicePsparsa = (j, string)
                print(indicePsparsa)
                listaBolada.append(indicePsparsa)
                break

    print("\n\n\n")
    print("INDICES", listaBolada)

    # a partir daqui não consegui testar pois não compilou em tempo habil
    iteradorlinhas = treino.shape()[0]
    listasfreqcomentario = list()
    listacolfreq = list()
    for i in range(iteradorlinhas):
        for j in listaBolada:
            # primeiro comentarios positivos
            if(iteradorlinhas < iteradorlinhas / 2)
                listacolfreq.append([treino[(i, j[0])], 1])
            # segundo cometnarios negativos
            else:
                listacolfreq.append([treino[(i, j[0]), 0])

    menoresdist = menoresdistancias(treino, listacolfreq, k)
    # vectorizer.get_feature_names()[freq_teste[]]

    # adc vet
    # vet for len(vet)
    #        roda dis euclidiana


if __name__ == "__main__":
    categorias = list()
    categorias.append("pos")
    categorias.append("neg")
    local = os.getcwd()
    dados = load_files(local, description=None, categories=categorias, load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)

    print("KNN")
    matrizSparsa = treino(dados)
    print(matrizSparsa)
    # print("\n", matrizSparsa)
    vectorizer = TfidfVectorizer()
    var = vectorizer.fit_transform(dados.data)
 #   teste = "I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge."

    cteste = list()
    cteste.append("cteste")
    teste = load_files(local, description=None, categories="classifica", load_content=True, shuffle=False, encoding="utf-8", decode_error='strict', random_state=0)
    vectorizerteste = TfidfVectorizer()
    varteste = vectorizerteste.fit_transform(teste.data)
    # print("teste", teste.data[1])
    # print("oi", vectorizerteste.get_feature_names())

    # print(varteste)
    # print("datamain", teste.data)

    predicao(var, varteste, 10, vectorizer, teste.data[0])
    # print(vectorizer.get_feature_names()[12343])
    # print("oi", matrizSparsa[(, 7684)])
    # print("alalalal")
    # print(matrizSparsa(0))
    # which(colnames(matrizSparsa) == "love")
    # list(matrizSparsa.columns.values)
    # predicao(matrizSparsa,comentario,3)
