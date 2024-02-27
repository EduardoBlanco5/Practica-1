import random
import numpy as np
import csv
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.1, max_epocas=1000):
        self.num_entradas = num_entradas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_epocas = max_epocas
        self.pesos = np.random.rand(num_entradas + 1)  # +1 para el sesgo

    def predict(self, entrada):
        suma_ponderada = np.dot(self.pesos[1:], entrada) + self.pesos[0]  # Producto punto + sesgo
        return 1 if suma_ponderada > 0 else 0

    def train(self, datos_entrenamiento):
        for _ in range(self.max_epocas):
            errores = 0
            for entrada, salida in datos_entrenamiento:
                prediccion = self.predict(entrada)
                error = salida - prediccion
                if error != 0:
                    errores += 1
                    self.pesos[1:] += self.tasa_aprendizaje * error * np.array(entrada)
                    self.pesos[0] += self.tasa_aprendizaje * error
            if errores == 0:
                break

def leer_datos(archivo):
    datos = []
    with open(archivo, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            entrada = list(map(float, row[:-1]))
            salida = int(row[-1])
            datos.append((entrada, salida))
    return datos

def mostrar_datos(datos, titulo):
    plt.figure()
    plt.title(titulo)
    for entrada, salida in datos:
        color = 'red' if salida == 1 else 'blue'
        plt.scatter(entrada[0], entrada[1], c=color)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.show()

# Lectura de datos de entrenamiento y prueba
datos_entrenamiento = leer_datos('XORtrn.csv')
datos_prueba = leer_datos('XORtst.csv')

# Mostrar datos de entrenamiento y prueba
mostrar_datos(datos_entrenamiento, 'Datos de Entrenamiento')
mostrar_datos(datos_prueba, 'Datos de Prueba')

# Entrenamiento del perceptrón
perceptron = Perceptron(num_entradas=2)
perceptron.train(datos_entrenamiento)

# Prueba del perceptrón entrenado
print("Predicciones después del entrenamiento:")
for entrada, _ in datos_prueba:
    prediccion = perceptron.predict(entrada)
    print(f"Entrada: {entrada}, Predicción: {prediccion}")