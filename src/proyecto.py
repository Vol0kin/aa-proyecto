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
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve, \
    cross_val_predict
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

def create_lr_pipeline(c_list):
    """
    Funcion para crear una lista de pipelines con el modelo
    de Logistic Regression dados unos valores de C, aplicando
    antes un escalado y PCA con 95% de varianza explicada

    :param c_list: Lista de valores C. Un valor por
                   cada LR del pipeline

    :return Devuelve una lista con los pipelines
    """

    # Crear lista de pipelines
    pipelines = []

    # Insertar nuevo pipeline
    for c in c_list:
        pipelines.append(
            make_pipeline(StandardScaler(), PCA(n_components=0.95),
                          LogisticRegression(C=c, random_state=1)))

    return pipelines


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


def create_rf_pipeline(n_estimators_list):
    """
    Funcion para crear una lista de pipelines con el modelo
    de Random Forest dados unos valores que determinan el numero de arboles,
    aplicando antes un escalado y PCA con 95% de varianza explicada

    :param n_estimators_list: Lista de valores para el nº de arboles.
                              Un valor por cada RF del pipeline

    :return Devuelve una lista con los pipelines
    """

    # Crear lista de pipelines
    pipelines = []

    # Insertar nuevo pipeline
    for n_estimators in n_estimators_list:
        pipelines.append(
            make_pipeline(StandardScaler(), PCA(n_components=0.95),
                          RandomForestClassifier(n_estimators=n_estimators, random_state=1)))

    return pipelines


def create_nn_pipeline(hidden_layer_sizes_list, early_stopping=False):
    """
    Funcion para crear una lista de pipelines con el modelo de Neural
    Network dados un valores que determinan el tamaño de las capas ocultas,
    aplicando antes un escalado y PCA con 95% de varianza explicada

    :param hidden_layer_sizes_list: Lista de valores para el nº de arboles.
                              Un valor por cada RF del pipeline

    :return Devuelve una lista con los pipelines
    """

    # Crear lista de pipelines
    pipelines = []

    # Insertar nuevo pipeline
    for hidden_layer_sizes in hidden_layer_sizes_list :
        pipelines.append(
            make_pipeline(StandardScaler(), PCA(n_components=0.95),
                          MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                        early_stopping=early_stopping, random_state=1)))

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

        plot_learning_curve(model, model_names[idx], X_train, y_train, cv=cv)

    return means, deviations


def prediction_evaluated_models(models, X, y, model_names, cv=10):
    for idx, model in enumerate(models):
        expected = y
        predicted = cross_val_predict(model, X, y, cv=cv)

        # plot_learning_curve(model, model_names[idx], X, y, cv=cv)

        print("\n\n",model_names[idx])
        print("Classification report:\n%s\n"
              % (metrics.classification_report(expected, predicted)))

        precision, recall, fscore, support = precision_recall_fscore_support(expected, predicted, average='macro')
        print('Precision : {}'.format(precision))
        print('Recall    : {}'.format(recall))
        print('F-score   : {}'.format(fscore))
        print('Support   : {}'.format(support))

        print("\n\nConfusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


##################################################################
# Visualizar resultados y graficas

def plot_pearson_correlation(data, fig_size):
    """
    Funcion para dibujar un grafico con la matriz con lo coeficientes de
    correlacion de Pearson
    
    :param data: Conjunto de datos de los que obtener la matriz de coeficientes
    :param fig_size: Escala de la figura a dibujar
    """
    
    # Obtener matriz de correlacion con 3 decimales
    corr = data.corr().round(3)

    # Establecer escala de figura y pintar
    plt.figure(figsize=fig_size)
    plt.title('Pearson Correlation Indexes Matrix')
    sns.heatmap(corr, vmin=-1, cmap='Spectral', annot=True, xticklabels=True, yticklabels=True)
    
    plt.show()


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
# Lectura de los datos
    
df1 = read_data_values('datos/segmentation.data')
df2 = read_data_values('datos/segmentation.test')
df = pd.concat([df1,df2])

#########################################################
# Crear mapa para cambiar los valores de las etiquetas

labels_to_values = { 'BRICKFACE' : 0, 'SKY' : 1, 'FOLIAGE' : 2, 'CEMENT' : 3,
                     'WINDOW' : 4, 'PATH' : 5, 'GRASS' : 6 }

# Sustituir etiquetas de salida por valores numericos discretos
df[0] = df[0].map(labels_to_values)


#########################################################
# Dividir en train y test

# Obtener valores X, Y
X, y = divide_data_labels(df)

# Dividir los datos en training y test
# Conservar proporcionalidad de clase
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1,
                                                    shuffle=True,
                                                    stratify=y)                # Estratificar 

# Crear lista con titulos de columnas (nombres de atributos y clase)
labels = ['RCL', 'RCR', 'RPC', 'SLD5', 'SLD2', 'VMean', 'VStd',
             'HMean', 'HStd', 'IntM', 'RRM', 'RBM', 'RGM', 'ERM',
             'EBM', 'EGM', 'ValM', 'SatM', 'HueM', 'Class']

# Crear DataFrame con los datos de training
train_df = pd.DataFrame(data=np.c_[X_train, y_train], columns=labels)

# Crear DataFrame con los datos de test
test_df = pd.DataFrame(data=np.c_[X_test, y_test], columns=labels)


#########################################################
# Obtener matriz de correlacion de Pearson


plot_pearson_correlation(train_df, (15, 15))

input('---Press any key to continue---\n\n')

#########################################################
# Eliminar variables con alta correlacion

# Crear lista de variables a eliminar
rm_list = [2, 10, 11, 12, 16]

X_train = np.delete(X_train, rm_list, axis=1)
X_test = np.delete(X_test, rm_list, axis=1)


input('---Press any key to continue---\n\n')



#########################################################
# Evaluar modelos

# Creamos las listas de valores para los determinados modelos
c_list = [0.1, 1.0, 5.0]
n_estimators_list = [10, 25, 50, 100]
hidden_layer_sizes_list = [10, 38, 100]
hidden_layer_sizes_list2 = [(10,10), (38,38), (100,100)]



# Asignamos nombres a los modelos
model_names = ['LR c = 0.1', 'LR c = 1.0', 'LR c = 5.0',
               'SVMC c = 0.1', 'SVMC c = 1.0', 'SVMC c = 5.0',
               'RF n_estimators = 10', 'RF n_estimators = 25', 'RF n_estimators = 50', 'RF n_estimators = 100',
               'NN hidden_layer_sizes = 10', 'NN hidden_layer_sizes = 38', 'NN hidden_layer_sizes = 100',
               'NN hidden_layer_sizes = 10-es', 'NN hidden_layer_sizes = 38-es', 'NN hidden_layer_sizes = 100-es']

# Crear 10-fold que conserva la proporcion
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Crear pipelines para cada modelo
lr_pipe = create_lr_pipeline(c_list)
svmc_pipe = create_svmc_pipeline(c_list)
rf_pipe = create_rf_pipeline(n_estimators_list)
nn_pipe = create_nn_pipeline(hidden_layer_sizes_list)
nn_pipe_es = create_nn_pipeline(hidden_layer_sizes_list2)

models = lr_pipe + svmc_pipe + rf_pipe + nn_pipe + nn_pipe_es

# Obtener valores medios, desviaciones y curvas de aprendizaje de los modelos
print('Evaluating models...')

means, deviations = evaluate_models(models, X_train, y_train,
                                    model_names, cv=cv, metric='accuracy')

# Mostrar valores por pantalla
print_evaluation_results(model_names, means, deviations, 'Mean Accuracy')


predicted_models_names = ['NN hidden_layer_sizes = 10', 'NN hidden_layer_sizes = 38',
                          'NN hidden_layer_sizes = 100']
predicted_models = nn_pipe
prediction_evaluated_models(predicted_models, X_test, y_test, predicted_models_names, cv)


input('---Press any key to continue---\n\n')

