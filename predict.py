# USO: Passar como argumento o diretório de imagens para treinamento e teste.
# python predict.py --training images/training --testing images/testing

# Importa as bibliotecas necessárias
import argparse
import cv2
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from patterns.localbinarypatterns import LocalBinaryPatterns
from scipy import interp
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn import metrics

# Seta debugger
from IPython.core.debugger import Tracer; debug_here = Tracer()

# Parseia os argumentos da linha de comando.
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
    help="path to the training images")
ap.add_argument("-e", "--testing", required=True,
    help="path to the tesitng images")
args = vars(ap.parse_args())

# Inicializa o descritor LBP (Local Binary Patterns)
# com os dados e a lista de labels
desc = LocalBinaryPatterns(24, 8)
data = [] # será armazedado para cada imagem sua descrição LBP.
labels = [] # será armazedado os rótulos para classficação.

# Itera sobre o conjunto de treinamento de imagens (training images)
for imagePath in paths.list_images(args["training"]):
    # Carrega a imagem, converte para escala cinza e descreve isso com LBP.
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gera as imagens LBP e Histograma
    hist = desc.describe(gray) # Retorna o resultado do descritor LBP.

    # Extrai os labels dos paths da imagem, depois atualiza a
    # lista de dados e labels inicializada anteriormente.
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)

# Resultado
# data = [hist_1, ... , ..., hist_8]
# labels = ['good', ..., 'bad']

# Cria dataframe e adiciona os labels
df = pd.DataFrame(data)
df["label"] = labels

# Separa dados e labels novamente
labels = df["label"]
data = df.iloc[:, list(range(0,26))]

# Embaralha a base
df = df.sample(frac=1).reset_index(drop=True)

# Separa base de dados em treinamento e testes (usa 30% para teste) - Use caso não queira validação cruzada.
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Treina um classificador SVM nos dados com diferentes parâmetros.
# Para maiores informações: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
cs = [400] # especificar parâmetros a serem testados
for c in cs:
    svm = LinearSVC(C=c, max_iter=30000, random_state = 0)
    scores = cross_val_score(svm, data, labels, cv=5)
    predictions = cross_val_predict(svm, data, labels, cv=5)
    print("SVM stats")
    print("Accuracy: ", metrics.accuracy_score(labels, predictions))
    print("Precision: ", metrics.precision_score(labels, predictions, pos_label='1'))
    print("Recall: ", metrics.recall_score(labels, predictions, pos_label='1'))
    print("F1 Score: ", metrics.f1_score(labels, predictions, pos_label='1'))
    print("ROC AUC Score: ", metrics.roc_auc_score(labels, predictions))
    print(metrics.confusion_matrix(labels, predictions))

# Treina um classificador GradientBoostingClassifier nos dados com diferentes parâmetros.
# Para maiores informações: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
learning_rates = [0.1] # especificar parâmetros a serem testados
max_depth = [10] # especificar parâmetros a serem testados
n_estimators = [240] # especificar parâmetros a serem testados
for lr in learning_rates:
    for md in max_depth:
        for est in n_estimators:
            gb = GradientBoostingClassifier(n_estimators=est, learning_rate = lr, max_depth = md, random_state = 0)
            scores = cross_val_score(gb, data, labels, cv=5)
            predictions = cross_val_predict(gb, data, labels, cv=5)
            print("GB Stats")
            print("Accuracy: ", metrics.accuracy_score(labels, predictions))
            print("Precision: ", metrics.precision_score(labels, predictions, pos_label='1'))
            print("Recall: ", metrics.recall_score(labels, predictions, pos_label='1'))
            print("F1 Score: ", metrics.f1_score(labels, predictions, pos_label='1'))
            print("ROC AUC Score: ", metrics.roc_auc_score(labels, predictions))
            print(metrics.confusion_matrix(labels, predictions))