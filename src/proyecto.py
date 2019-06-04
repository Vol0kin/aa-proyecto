# -*- coding: utf-8 -*-
"""
PROYECTO FINAL
Autores: Vladislav Nikolov Vasilev
         Jose Maria Sanchez Guerrero
"""

# Modulos generales
import numpy as np
import pandas as pd

# Modulos para graficos
import seaborn as sns
import matplotlib.pyplot as plt

# Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

# Metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Utiles para seleccion de modelos
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve

# Pipelines
from sklearn.pipeline import make_pipeline

# Grid Search
from sklearn.model_selection import GridSearchCV

# Ignorar warnings
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
            make_pipeline(LogisticRegression(C=c, multi_class='multinomial',
                                             solver='newton-cg', random_state=1)))

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
            make_pipeline(StandardScaler(),
                          SVC(C=c, random_state=1, gamma='auto')))

    return pipelines


def create_rf_pipeline(n_estimators_list, max_depth=None):
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
            make_pipeline(RandomForestClassifier(n_estimators=n_estimators,
                                                 max_depth=max_depth, random_state=1)))

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
            make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                        early_stopping=early_stopping, random_state=1)))

    return pipelines



##################################################################
# Evaluacion de modelos

def evaluate_models(models, X, y, model_names, cv=10, metric='accuracy',
                    plot_learn=False):
    """
    Funcion para evaluar un conjunto de modelos con un conjunto
    de caracteristicas y etiquetas.


    :param models: Lista que contiene pipelines o modelos que
                   se quieren evaluar
    :param X: Conjunto de datos con los que evaluar
    :param y: Conjunto de etiquetas
    :param cv: Numero de k-folds que realizar (por defecto 10)
    :param metric: Metrica de evaluacion (por defecto es la precision, accuracy)
    :param plot_learn: Indica si dibujar las curvas de aprendizaje (por defecto
                       False)

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

        # Imprime la curva de aprendizaje del modelo
        if plot_learn:
            plot_learning_curve(model, model_names[idx], X_train, y_train, cv=cv)

    return means, deviations


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
    sns.heatmap(corr, vmin=-1, cmap='Spectral', annot=True, xticklabels=True,
                yticklabels=True)
    
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


def plot_learning_curve(model, title, X, y, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Funcion para mostrar una gráfica con la curva de aprendizaje

    :param model: Modelo que se quiere evaluar
    :param X: Conjunto de datos con los que evaluar
    :param y: Conjunto de etiquetas
    :param cv: Numero de k-folds que realizar (por defecto 10)
    :param train_sizes: Número de evaluaciones o puntos que tendrá
                        la curva de validación
    """

    # Establecer escala de figura y títulos
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # Calcula la curva de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=None, train_sizes=train_sizes)

    # Calcula la media de aciertos y desviación típica
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Genera las áreas de influencia de las dos rectas
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Genera las rectas de para el training y el test
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # Imprime la curva
    plt.legend(loc="best")
    plt.grid()
    plt.show()


#########################################################
# Lectura de los datos

print('IMAGE SEGMENTATION DATA SET\n')
print('Reading data...')

df1 = read_data_values('datos/segmentation.data')
df2 = read_data_values('datos/segmentation.test')
df = pd.concat([df1,df2])

print('Data read!')

input('\n---Press any key to continue---\n\n')

#########################################################
# Crear mapa para cambiar los valores de las etiquetas

print('Converting class labels to numeric values...')
classes = ['BRICKFACE', 'SKY', 'FOLIAGE', 'CEMENT', 'WINDOW', 'PATH', 'GRASS']
labels_to_values = { 'BRICKFACE' : 0, 'SKY' : 1, 'FOLIAGE' : 2, 'CEMENT' : 3,
                     'WINDOW' : 4, 'PATH' : 5, 'GRASS' : 6 }

print('Mapping: {}'.format(labels_to_values))

# Sustituir etiquetas de salida por valores numericos discretos
df[0] = df[0].map(labels_to_values)

input('\n---Press any key to continue---\n\n')

#########################################################
# Dividir en train y test

# Obtener valores X, Y
print('Getting X and y values from data...')
X, y = divide_data_labels(df)

# Dividir los datos en training y test
# Conservar proporcionalidad de clase
print('Splitting data in training and test sets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1,
                                                    shuffle=True,
                                                    stratify=y)

# Crear lista con titulos de columnas (nombres de atributos y clase)
labels = ['RCL', 'RCR', 'RPC', 'SLD5', 'SLD2', 'VMean', 'VStd',
          'HMean', 'HStd', 'IntM', 'RRM', 'RBM', 'RGM', 'ERM',
          'EBM', 'EGM', 'ValM', 'SatM', 'HueM', 'Class']

# Crear DataFrame con los datos de training
train_df = pd.DataFrame(data=np.c_[X_train, y_train], columns=labels)

# Crear DataFrame con los datos de test
test_df = pd.DataFrame(data=np.c_[X_test, y_test], columns=labels)


# Imprimimos un ejemplo del dataset
print('Training data sample:')
print(train_df.head())

# Determinar numero de muestras por conjunto
print('Training data size: ', train_df.shape[0])
print('Test data size: ', test_df.shape[0])

# Determinar si faltan valores para los dos conjunts
print('Missing values in train? ', train_df.isnull().values.any())
print('Missing values in test? ', test_df.isnull().values.any())

input('\n---Press any key to continue---\n\n')

#########################################################
# Obtener matriz de correlacion de Pearson
plot_pearson_correlation(train_df, (15, 15))

input('\n---Press any key to continue---\n\n')

#########################################################
# Eliminar variables con alta correlacion

print('Removing correlated variables from training and test...')

# Crear lista de variables a eliminar
rm_list = [2, 10, 11, 12, 16]

X_train = np.delete(X_train, rm_list, axis=1)
X_test = np.delete(X_test, rm_list, axis=1)

print('Removal complete!')
input('\n---Press any key to continue---\n\n')



#########################################################
# Evaluar modelos

print('Preparing data for models evaluation...')

# Creamos las listas de valores para los determinados modelos
c_list = [0.1, 1.0, 5.0]
n_estimators_list = [10, 25, 50, 100]
hidden_layer_sizes_list = [10, 38, 100, (10,10), (38,38), (100,100),
                           (10,10,10), (38,38,38), (100,100,100)]

# Asignamos nombres a los modelos
model_names = ['LR c = 0.1', 'LR c = 1.0', 'LR c = 5.0',
               'SVMC c = 0.1', 'SVMC c = 1.0', 'SVMC c = 5.0',
               'RF n_estimators = 10', 'RF n_estimators = 25',
               'RF n_estimators = 50', 'RF n_estimators = 100',
               'RF n_estimators = 10, max_depth = 9',
               'RF n_estimators = 25, max_depth = 9',
               'RF n_estimators = 50, max_depth = 9',
               'RF n_estimators = 100, max_depth = 9',
               'NN hidden_layer_sizes = 10', 'NN hidden_layer_sizes = 38',
               'NN hidden_layer_sizes = 100', 'NN hidden_layer_sizes = 10-10',
               'NN hidden_layer_sizes = 38-38',
               'NN hidden_layer_sizes = 100-100',
               'NN hidden_layer_sizes = 10-10-10',
               'NN hidden_layer_sizes = 38-38-38',
               'NN hidden_layer_sizes = 100-100-100']

# Crear 10-fold que conserva la proporcion
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Crear pipelines para cada modelo
lr_pipe = create_lr_pipeline(c_list)
svmc_pipe = create_svmc_pipeline(c_list)
rf_pipe = create_rf_pipeline(n_estimators_list)
rf_pipe_depth = create_rf_pipeline(n_estimators_list, 9)
nn_pipe = create_nn_pipeline(hidden_layer_sizes_list)

models = lr_pipe + svmc_pipe + rf_pipe + rf_pipe_depth + nn_pipe

# Obtener valores medios, desviaciones y curvas de aprendizaje de los modelos
print('Evaluating models...')

means, deviations = evaluate_models(models, X_train, y_train, model_names,
                                    cv=cv, plot_learn=False)

# Mostrar valores por pantalla
print_evaluation_results(model_names, means, deviations, 'Mean Accuracy')

input('\n---Press any key to continue---\n\n')

#########################################################
# Ajuste de hiperparámetros

print('Tunning hyperparameters for Multinomial Logistic Regression...')

# Crear grid de hiperparametros que se van a probar
print('Creating parameters grid...')
param_grid_lr = [{'C': np.linspace(0.1, 1.0, 10),
                  'multi_class': ['multinomial'],
                  'solver': ['newton-cg'],
                  'random_state': [1]}]

# Crear modelo de regresion logistica
print('Creating model...')
mlr = LogisticRegression()

# Aplicar GridSearch con Cross Validation para determinar
# la mejor combinacion de parametros
grid_search = GridSearchCV(mlr, param_grid=param_grid_lr, cv=cv,
                           scoring='accuracy')

# Aplicar GridSearch para obtener la mejor combinacion de hiperparametros
print('Applying grid search...')
grid_search.fit(X_train, y_train)

# Mostrar informacion sobre el ajuste de hiperparametros 
print('Grid search complete! Showing results below\n')
print(grid_search.best_estimator_)

# Obtener indice de la mejor media de test
best_idx = np.argmax(grid_search.cv_results_['mean_test_score'])

print('\nMean training accuracy: {}'.format(grid_search.cv_results_['mean_train_score'][best_idx]))
print('Training accuracy Std. Dev: {}'.format(grid_search.cv_results_['std_train_score'][best_idx]))
print('Mean CV accuracy: {}'.format(grid_search.cv_results_['mean_test_score'][best_idx]))
print('CV accuracy Std. Dev: {}'.format(grid_search.cv_results_['std_test_score'][best_idx]))

input('\n---Press any key to continue---\n\n')
###############################################################################

print('Tunning hyperparameters for Random Forest...')

# Crear grid de hiperparametros que se van a probar
print('Creating parameters grid...')
param_grid_rf = [{'n_estimators': np.linspace(100, 500, 9, dtype=np.int),
                  'max_depth': np.linspace(5, 15, 6, dtype=np.int),
                  'random_state': [1]}]

# Crear modelo de Random Forest
print('Creating model...')
rf = RandomForestClassifier()

grid_search2 = GridSearchCV(rf, param_grid=param_grid_rf, cv=cv,
                            scoring='accuracy')

# Aplicar GridSearch para obtener la mejor combinacion de hiperparametros
print('Applying grid search...')
grid_search2.fit(X_train, y_train)

# Mostrar informacion sobre el ajuste de hiperparametros
print('Grid search complete! Showing results below\n')
print(grid_search2.best_estimator_)

# Obtener indice de la mejor media de test
best_idx = np.argmax(grid_search2.cv_results_['mean_test_score'])

print('\nMean training accuracy: {}'.format(grid_search2.cv_results_['mean_train_score'][best_idx]))
print('Training accuracy Std. Dev: {}'.format(grid_search2.cv_results_['std_train_score'][best_idx]))
print('Mean CV accuracy: {}'.format(grid_search2.cv_results_['mean_test_score'][best_idx]))
print('CV accuracy Std. Dev: {}'.format(grid_search2.cv_results_['std_test_score'][best_idx]))

input('\n---Press any key to continue---\n\n')

#########################################################
# Evaluación del mejor modelo comparandolo con modelo de referencia

print('Creating dummy classifier and predicting labels...')

# Creamos modelo de prueba y lo entrenamos
dummy = DummyClassifier()
dummy.fit(X_train, y_train)

# Predecir valores con dummy
y_predicted_dummy = dummy.predict(X_test)

print('Predicting labels with Random Forest...')

# Asignamos un nuevo nombre al GridSearch del RF (se puede usar para predecir)
rf_model = grid_search2

# Predecimos valores con RF
y_predicted_rf = rf_model.predict(X_test)

# Obtener metricas
dummy_score = accuracy_score(y_test, y_predicted_dummy)
rf_score = accuracy_score(y_test, y_predicted_rf)

dummy_recall = recall_score(y_test, y_predicted_dummy, average='macro')
rf_recall = recall_score(y_test, y_predicted_rf, average='macro')

dummy_precision = precision_score(y_test, y_predicted_dummy, average='macro')
rf_precision = precision_score(y_test, y_predicted_rf, average='macro')

print('Dummy classifier accuracy score: ', dummy_score)
print('Random Forest accuracy score: ', rf_score)
print('Dummy classifier recall score: ', dummy_recall)
print('Random Forest recall score: ', rf_recall)
print('Dummy classifier precision score: ', dummy_precision)
print('Random Forest precision score: ', rf_precision)

input('\n---Press any key to continue---\n\n')

# Matriz de confusion
matrix = confusion_matrix(y_test, y_predicted_rf)

# Crear DataFrame que pintar
confusion_mat = pd.DataFrame(data=matrix, index=classes, columns=classes)

plt.title('Confusion Matrix for Random Forest model')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
sns.heatmap(confusion_mat,annot=True)
plt.show()

input('\n---Press any key to continue---\n\n')

# Curvas de aprendizaje de Random Forest
plot_learning_curve(RandomForestClassifier(n_estimators=150, max_depth=13, random_state=1),
                    title='Random Forest n_estiamtors = 100 Depth = 13 levels',
                    X=X_train, y=y_train, cv=cv)

input('\n---Press any key to continue---\n\n')
