
"""
Created on Sun Dec  3 13:10:22 2017

@author: Marco
"""
 


from sklearn.datasets import*
from sklearn.feature_extraction.text import *

categorias = [ 'Negativo' , 'Positivo']
dados = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\ComentariosIMDB", description = None , categories = categorias , load_content = True , shuffle = True  , decode_error = 'strict' , random_state = 0 )

indiceNegativos = list()

for i in range (len(dados.target)):
    if dados.target[i] == 0:
        indiceNegativos.append(i)



count_vect = CountVectorizer() # Cria a Sacola de Palavras
dados_counts = count_vect.fit_transform(dados.data) #

#print(dados_counts)
#print(count_vect.vocabulary_.get('great'))


tf_transformer = TfidfTransformer(use_idf = False).fit(dados_counts)
dados_tf = tf_transformer.transform(dados_counts)


tfidf_transformer = TfidfTransformer()
dados_tfidf = tfidf_transformer.fit_transform(dados_counts)

print(dados_tfidf)
print("-----------------------------------------------------------------------------------------------")
#print(dados_tf)

feature_names = count_vect.get_feature_names()

#print(len(feature_names))
#print(feature_names)
#print(feature_names[73821])




