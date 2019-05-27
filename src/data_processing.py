# -*- coding: utf-8 -*-
"""
PROYECTO FINAL
Autores: Vladislav Nikolov Vasilev
         Jose Maria Sanchez Guerrero
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Fijamos la semilla
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



######################################
# Division en train y test

df = read_data_values('datos/segmentation.data')
X_train, y_train = divide_data_labels(df)

df = read_data_values('datos/segmentation.test')
X_test, y_test = divide_data_labels(df)


# Crear DataFrame con los datos de training
train_df = pd.DataFrame(data=np.c_[X_train, y_train])

# Crear DataFrame con los datos de test
test_df = pd.DataFrame(data=np.c_[X_test, y_test])

input('---Press any key to continue---\n\n')

