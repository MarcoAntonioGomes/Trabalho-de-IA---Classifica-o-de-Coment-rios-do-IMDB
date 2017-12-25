from sklearn.datasets import *
from sklearn.neighbors import *
from collections import Counter
from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

categorias = ["positivo", "negativo"]

#Fazendo o carregamento dos arquivos que contém os comentários
dados = load_files("E:\\Engenharia de Computação\\6º Semestre\\Inteligência artificial\\Trabalho final\\ComentariosIMDB", description = "Testando essa porra", categories = categorias, load_content = True, shuffle = False, encoding = "utf-8", decode_error = 'strict', random_state = 0)

#Conversão dos documentos de texto em uma matriz de contagem
vec=CountVectorizer()

#Preparação e aprendizado dos dados
fit = vec.fit_transform(dados.data)

X_treino, X_teste, y_treino, y_teste = train_test_split(fit, dados.target, test_size=0.01, random_state=0)



#Classificador com constante igual a 5 (numéro de vizinhos a ser considerados a partir do dado que vai ser classificado.
#Impacto médio na porcentagem de acerto final do algoritmo
neigh = KNeighborsClassifier(n_neighbors = 5)

neigh.fit(X_treino, y_treino)

neigh.set_params(p=7)

neigh.predict(X_teste)

#Mostra A porcentagem de acerto dos comentários testados em função da base teste
print(neigh.score(X_teste,y_teste))

