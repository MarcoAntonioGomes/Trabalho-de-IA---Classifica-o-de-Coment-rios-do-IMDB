from sklearn.datasets import *
from sklearn.neighbors import *
from collections import Counter
from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


categorias = ["positivo", "negativo"]

dados = load_files("C:\\ComentariosIMDB", description = "Testando essa porra", categories = categorias, load_content = True, shuffle = False, encoding = "utf-8", decode_error = 'strict', random_state = 0)

vec=CountVectorizer()
fit = vec.fit_transform(dados.data)

X_train, X_test, y_train, y_test = train_test_split(fit, dados.target, test_size=0.1, random_state=0)




neigh = KNeighborsClassifier(n_neighbors = 12)





neigh.fit(X_train, y_train)

neigh.set_params(p=7)


neigh.predict(X_test)


print(neigh.score(X_test,y_test))