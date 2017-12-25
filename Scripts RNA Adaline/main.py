from sklearn.datasets import *
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB
from threading import Thread


categorias = ["positivo", "negativo"]
dados = load_files("C:\\Users\\Marco\\Desktop\\6º Periodo\\Inteligencia Artificial\\ComentariosIMDB", description=None,
                   categories=categorias, load_content=True, shuffle=False, decode_error='strict', random_state=0)

comentarios = dados.data
vetorizador = CountVectorizer()
comentarios_fit = vetorizador.fit_transform(dados.data)

sacola = vetorizador.get_feature_names()
frequenciaPositiva = []
frequenciaNegativa = []

soma_1 = [] #Soma Positivo  0 a 3127
soma_2 = [] #Soma Positivo  3127 a 6248
soma_3 = [] #Soma Positivo  6248 a 9372
soma_4 = [] #Soma Positivo  9372 s 12499

soma_5 = [] # Soma  Negativo 12500 a 15624
soma_6 = [] # Soma  Negativo 15624 a 18748
soma_7 = [] # Soma  Negativo 18748 a 21872
soma_8 = [] # Soma  Negativo 21872 a 24999

interval_1 = 0
interval_2 = 3127
interval_3 = 6248
interval_4 = 9372
interval_5 = 12499


interval_6 = 12500
interval_7 = 15624
interval_8 = 18748
interval_9 = 21872
interval_10 = 24999




class  preProcess(Thread):

    def __init__(self, interval_1,interval_2,interval_3,interval_4, soma_1,soma_2, verifica):
        Thread.__init__(self)
        self.interval_1 = interval_1
        self.interval_2 = interval_2
        self.interval_3 = interval_3
        self.interval_4 = interval_4
        self.soma_1 = soma_1
        self.soma_2 = soma_2
        self.verifica = verifica



    def run(self):

        for i in range(len(sacola)):
            soma = 0
            for j in range(self.interval_1,self.interval_2):
                soma = soma + comentarios_fit[(j,i)]
                if(self.verifica):
                    print("comentarios_fit[(", j, ",", i, ")]")
            self.soma_1.append(soma)

        for i in range(len(sacola)):
            soma = 0
            for j in range(self.interval_3, self.interval_4):
                soma = soma + comentarios_fit[(j, i)]
                #print("!!comentarios_fit[(", j, ",", i, ")]")
            self.soma_2.append(soma)


t1 = preProcess(interval_1,interval_2,interval_6,interval_7,soma_1,soma_5, True)
t2 = preProcess(interval_2,interval_3,interval_7,interval_8,soma_2,soma_6,False)
t3 = preProcess(interval_3,interval_4,interval_8,interval_9,soma_3,soma_7,False)
t4 = preProcess(interval_4,interval_5,interval_9,interval_10,soma_4,soma_8,False)

t1.start()
t2.start()
t3.start()
t4.start()










# print(sacola)
#
# for i in range (len(sacola)):
#     soma = 0
#     for j in range (12499):
#         soma = soma + comentarios_fit[(j, i)]
#         print("comentarios_fit[(",j,",",i,")]")
#     frequenciaPositiva.append(soma)
#
# for i in range(len(sacola)):
#     soma = 0
#     for j in range(12500, 24999):
#         soma = soma + comentarios_fit[(j, i)]
#         print("!!comentarios_fit[(", j, ",", i, ")]")
#     frequenciaNegativa.append(soma)










