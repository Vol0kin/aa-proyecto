# -*- coding: utf-8 -*-
"""
PROYECTO FINAL
Autores: Vladislav Nikolov Vasilev
         Jose Maria Sanchez Guerrero
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings('ignore')


# Fijamos la semilla
from sklearn.svm import SVC

np.random.seed(1)



##################################################################
# Lectura y division de los datos

def read_data_values(in_file, separator=None):
    """
    Funcion para leer los datos de un archivo

    :param in_file Archivo de entrada
    :param separator Separador que se utiliza en el archivo (por defecto
                     None)

    :return Devuelve los datos leidos del archivo en un DataFrame
    """

    # Cargar los datos en un DataFrame
    # Se indica que la primera columna no es el header
    if separator == None:
        df = pd.read_csv(in_file, header=None, skiprows=5)
    else:
        df = pd.read_csv(in_file, sep=separator, header=None, skiprows=5)
    return df


def divide_data_labels(input_data):
    """
    Funcion que divide una muestra en los datos y las etiquetas

    :param input_data Conjunto de valores que se quieren separar
                      juntados en un DataFrame

    :return Devuelve los datos y las etiquetas
    """

    # Obtener los valores
    values = input_data.values

    # Obtener datos y etiquetas
    X = values[:, 1:]
    y = values[:, 0]

    return X, y



##################################################################
# Funciones para crear los pipelines de cada uno de los modelos

def create_svmc_pipeline(c_list):
    """
    Funcion para crear una lista de pipelines con el
    modelo de SVM dados unos valores de C, aplicando
    antes un escalado y PCA con 95% de varianza explicada

    :param c_list: Lista de valores C. Un valor por
                   cada SVM del pipeline

    :return Devuelve una lista con los pipelines
    """

    # Crear lista de pipelines
    pipelines = []

    # Insertar nuevo pipeline
    for c in c_list:
        pipelines.append(
            make_pipeline(StandardScaler(), PCA(n_components=0.95),
                          SVC(C=c, random_state=1, gamma='auto')))

    return pipelines


def create_svmc_pipeline2(c_list):
    """
    Funcion para crear una lista de pipelines con el
    modelo de SVM dados unos valores de C, aplicando
    antes un escalado y PCA con 95% de varianza explicada

    :param c_list: Lista de valores C. Un valor por
                   cada SVM del pipeline

    :return Devuelve una lista con los pipelines
    """

    # Crear lista de pipelines
    pipelines = []

    # Insertar nuevo pipeline
    for c in c_list:
        pipelines.append(
            make_pipeline(PCA(n_components=0.95), StandardScaler(),
                          SVC(C=c, random_state=1, gamma='auto')))

    return pipelines



##################################################################
# Evaluacion de modelos

def evaluate_models(models, X, y, model_names, cv=10, metric='neg_mean_absolute_error'):
    """
    Funcion para evaluar un conjunto de modelos con un conjunto
    de caracteristicas y etiquetas.


    :param models: Lista que contiene pipelines o modelos que
                   se quieren evaluar
    :param X: Conjunto de datos con los que evaluar
    :param y: Conjunto de etiquetas
    :param cv: Numero de k-folds que realizar (por defecto 10)
    :param metric: Metrica de evaluacion (por defecto es la norma-l1,
                   neg_mean_absolute_error)

    :return Devuelve una lista con los valores medios y una lista
            con las desviaciones tipicas
    """

    # Crear listas de medias y desviacions
    means = []
    deviations = []

    # Para cada modelo, obtener los resultados de
    # evaluar el modelo con todas las particiones
    # Guardar los resultados en las listas correspondientes
    for idx,model in enumerate(models):
        results = cross_val_score(model, X, y, scoring=metric, cv=cv)

        # Guardar valor medio de los errores
        # Se guarda el valor absoluto porque son valores negativos
        means.append(abs(results.mean()))

        # Guardar desviaciones
        deviations.append(np.std(results))

        plot_learning_curve(model, model_names[idx], X, y, cv=cv)


    return means, deviations



##################################################################
# Visualizar resultados de eval. y graficas de clases

def print_evaluation_results(models, means, deviations, metric):
    """
    Funcion para mostrar por pantalla los resultados
    de la evaluacion

    :param models: Nombres de los modelos evaluados
    :param means: Lista con los valores medios de las
                  evaluaciones
    :param deviations: Lista con los valores de las
                       desv. tipicas de las evaluaciones
    :param metric: Metrica a evaluar
    """

    print('Evaluation results for each model')

    # Crear un DataFrame con el formato de salida
    out_df = pd.DataFrame(index=models, columns=[metric, 'Standard Deviation'],
                          data=[[mean, dev] for mean, dev in zip(means, deviations)])
    print(out_df)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()



#########################################################
# Lectura y previsualizacion inicial de los datos
df1 = read_data_values('datos/segmentation.data')
df2 = read_data_values('datos/segmentation.test')
df = pd.concat([df1,df2])
# print(df)



#########################################################
# Dividir en train y test

# Obtener valores X, Y
X, y = divide_data_labels(df)

# Dividir los datos en training y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1, shuffle=True)

# Crear DataFrame con los datos de training
train_df = pd.DataFrame(data=np.c_[X_train, y_train])

# Crear DataFrame con los datos de test
test_df = pd.DataFrame(data=np.c_[X_test, y_test])



#########################################################
# Creamos mapa para cambiar los valores de las etiquetas
labels_to_values = { 'BRICKFACE' : 0, 'SKY' : 1, 'FOLIAGE' : 2, 'CEMENT' : 3,
                     'WINDOW' : 4, 'PATH' : 5, 'GRASS' : 6 }

# Aplicamos el mapa a las etiquetas
df[0] = df[0].map(labels_to_values)

input('---Press any key to continue---\n\n')



#########################################################
# Evaluar modelos

# Creamos las listas de valores para los determinados modelos
c_list = [0.1, 1.0, 5.0]

# Asignamos nombres a los modelos
model_names = ['SVMC N+PCA c = 0.1', 'SVMC N+PCA c = 1.0', 'SVMC N+PCA c = 5.0',
               'SVMC PCA+N c = 0.1', 'SVMC PCA+N c = 1.0', 'SVMC PCA+N c = 5.0']

# Crear 10-fold que conserva la proporcion
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Crear pipelines para cada modelo
svmc_pipe = create_svmc_pipeline(c_list)
svmc_pipe2 = create_svmc_pipeline2(c_list)

models = svmc_pipe + svmc_pipe2

# Obtener valores medios, desviaciones y curvas de aprendizaje de los modelos
print('Evaluating models...')

means, deviations = evaluate_models(models, X_train, y_train,
                                    model_names, cv=cv, metric='accuracy')

# Mostrar valores por pantalla
print_evaluation_results(model_names, means, deviations, 'Mean Accuracy')

input('---Press any key to continue---\n\n')

