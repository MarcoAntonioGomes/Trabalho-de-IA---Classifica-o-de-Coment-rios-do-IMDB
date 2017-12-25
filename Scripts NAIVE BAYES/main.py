from sklearn.datasets import *
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.model_selection import train_test_split

categorias = ["positivo", "negativo"]
dados = load_files("H:\\downloads\\ComentariosIMDB", description = None, categories = categorias, load_content = True, shuffle = False, encoding="utf-8", decode_error = 'strict', random_state = 0)

vetorizador = CountVectorizer()
vetorizador_invertido = TfidfTransformer()
dados_vetorizados = vetorizador.fit_transform(dados.data)
dados_invertidos = vetorizador_invertido.fit_transform(dados_vetorizados)
dados_treinamento, dados_teste, categorias_treinamento, categorias_teste = train_test_split(dados_invertidos, dados.target, test_size=0.1, random_state=0)
naive_bayes = MultinomialNB().fit(dados_invertidos, dados.target)
previsao = naive_bayes.predict(dados_teste)

print(naive_bayes.score(dados_teste, categorias_teste))