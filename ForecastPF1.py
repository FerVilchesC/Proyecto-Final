# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 20:08:06 2025
@author: VILCHFE



El código es para una aplicación con interfaz gráfica que permite seleccionar 
un archivo Excel con datos históricos de ingresos semanales en CLP, USD y EUR.
Luego, procesa esos datos agrupándolos por semanas.
 Usa esos datos para entrenar un modelo LSTM que aprende a 
 predecir los ingresos de las próximas tres semanas para las tres monedas. 
 La aplicación muestra las predicciones directamente en la ventana y
 ofrece la opción de guardar los resultados en un archivo Excel. 
 



"""

#%%

import os # Permite interactuar con el sistema operativo: acceder a rutas, listar archivos, crear/eliminar carpetas
import hashlib #Genera huellas digitales (hashes) como MD5 o SHA para verificar integridad de datos o identificar archivos
import tensorflow as tf #Framework para crear y entrenar modelos de inteligencia artificial y redes neuronales.
tf.random.set_seed(123)  # Semilla para reproducibilidad
import numpy as np # Maneja arreglos y operaciones matemáticas rápidas y eficientes.
np.random.seed(123) #Fija la semilla de aleatoriedad de NumPy para reproducir resultados.

from tensorflow.keras.models import Sequential # Crea modelos de redes neuronales de forma secuencial.
from tensorflow.keras.layers import LSTM, Dense

#LSTM: Capa de memoria a largo plazo, útil para datos secuenciales como series temporales.
#Dense: Capa totalmente conectada,  para la salida o capas intermedias.

from tkinter import Tk, Label, Button, Text, END, filedialog, messagebox #Interfax gráfica
from sklearn.preprocessing import MinMaxScaler #Normaliza o escala valores numéricos a un rango definido, para mejorar el rendimiento del modelo.
from datetime import timedelta #Maneja intervalos de tiempo, permitiendo sumarlos o restarlos a fechas.
import pandas as pd #Maneja y analiza DataFrames, ideal para leer, limpiar y procesar datos.

# Forzar operaciones determinísticas
os.environ['TF_DETERMINISTIC_OPS'] = '1' #Fuerza a TensorFlow para obtener siempre los mismos resultados.
os.environ['PYTHONHASHSEED'] = '0' #Fija la semilla del algoritmo de hash de Python para eliminar variaciones aleatorias.

#%% Obtener "huella digital" del archivo

def calcular_hash_archivo(ruta):
    """Calcula el hash MD5 de un archivo para detectar cambios"""
    hash_md5 = hashlib.md5() ## Crea objeto para calcular MD5
    with open(ruta, "rb") as f: ## Abre el archivo en modo binario
        for chunk in iter(lambda: f.read(4096), b""): ## Lee en bloques de 4096 bytes
            hash_md5.update(chunk) ## Actualiza el cálculo con cada bloque
    return hash_md5.hexdigest() #  # Devuelve el hash en formato hexadecimal


#%% Inicia ventana en TKinter y crea botones

class PrediccionApp:
    def __init__(self, root):
        """Inicializa la ventana Tkinter y sus botones"""
        self.root = root
        self.root.title("Predicción de Ingresos Semanales")
        self.root.geometry("800x520")

        self.archivo = None
        self.resultado = None
        self.modelo_path = "modelo_completo.h5" #Ruta del modelo guardado
        self.hash_path = "ultimo_hash.txt" #Ruta del archivo con último hash

        # Título
        Label(root, text="Predicción de Ingresos por Moneda (CLP, USD, EUR)", font=("Arial", 14)).pack(pady=10)

        # Botones
        Button(root, text="1. Seleccionar Archivo Excel", command=self.seleccionar_archivo).pack(pady=5)
        Button(root, text="2. Predecir", command=self.procesar).pack(pady=5)
        Button(root, text="3. Guardar Resultado en Excel", command=self.guardar_excel).pack(pady=5)

        # Área de texto para mostrar resultados
        Label(root, text="Resultados:").pack(pady=10)
        self.resultado_texto = Text(root, height=12, width=90)
        self.resultado_texto.pack()

        # Etiqueta de estado "Procesando..."
        self.label_procesando = Label(root, text="", fg="#993300", font=("Arial", 10, "italic"))
        self.label_procesando.pack(pady=5)

    def seleccionar_archivo(self):
        #Permite al usuario seleccionar el archivo Excel
        
        self.archivo = filedialog.askopenfilename(
            title="Selecciona archivo Excel",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if self.archivo:
            messagebox.showinfo("Archivo Seleccionado", f"Archivo cargado:\n{self.archivo}")
        else:
            messagebox.showwarning("Advertencia", "No se seleccionó ningún archivo.")
#%% Crear modelo y procesar


    def crear_modelo_lstm(self, input_shape):
        """Crea un modelo LSTM simple"""
        model = Sequential() #Modelo secuencial 
        model.add(LSTM(50, activation='relu', input_shape=input_shape)) # # Capa LSTM con 50 neuronas
        model.add(Dense(9))  # 3 semanas * 3 monedas
        model.compile(optimizer='adam', loss='mse') #Configura el entrenamiento.
        return model

    def procesar(self):
        """Procesa el archivo seleccionado, entrena/carga modelo y predice ingresos"""
        if not self.archivo: #Si no hay archivo seleccionado
        
            messagebox.showwarning("Advertencia", "Primero debes seleccionar un archivo.")
            return

        self.label_procesando.config(text="Procesando...") #Muestra el estado
        self.root.update_idletasks()

        try:
            # Calcular hash del archivo para detectar cambios
            hash_actual = calcular_hash_archivo(self.archivo) 
            hash_guardado = None
            if os.path.exists(self.hash_path): #Si hay archivo guardado:
                with open(self.hash_path, "r") as f:
                    hash_guardado = f.read()

            #  Leer Excel y seleccionar columnas necesarias
            df = pd.read_excel(self.archivo)
            df = df[['Date', 'CLP', 'USD', 'EUR', 'Day', 'Month', 'Week']].dropna()  # Selecciona columnas y Elimina filas incompletas
            df['Date'] = pd.to_datetime(df['Date'])# Convierte a formato fecha
            df = df.sort_values('Date') # Ordena por fecha

            #  Agrupar ingresos por semana (suma semanal)
            df = df.groupby(pd.Grouper(key='Date', freq='W-MON')).sum().reset_index()

            # Seleccionar columnas para el modelo 
            cols_modelo = ['CLP', 'USD', 'EUR', 'Day', 'Month', 'Week']
            datos = df[cols_modelo].values #  Convierte a array NumPy

            # Normalizar todas las columnas 
            scaler = MinMaxScaler()
            datos_escalados = scaler.fit_transform(datos) ## Escala valores al rango (0,1)

            #  Crear ventanas de entrenamiento para LSTM 
            ventana = 12  # Se Usan las últimas 12 semanas como entrada
            X, y = [], [] # Listas para datos de entrenamiento
            for i in range(len(datos_escalados) - ventana - 2): # Recorre datos creando ventanas
                X.append(datos_escalados[i:i + ventana]) # Entrada: 12 semanas
                y.append(datos_escalados[i + ventana:i + ventana + 3, :3].flatten())  #  Salida: 3 semanas de monedas

            X = np.array(X) # Convierte X a array
            y = np.array(y) # Convierte y a array

            # Cargar modelo si no hay cambios
            model = None
            if hash_actual == hash_guardado and os.path.exists(self.modelo_path):  # Si archivo no cambió y modelo existe
                try:
                    model = tf.keras.models.load_model(self.modelo_path) # Carga modelo desde disco
                    messagebox.showinfo("Información", "Archivo sin cambios. Modelo cargado desde disco.")
                except Exception as e:
                    messagebox.showwarning(
                        "Advertencia",
                        f"Modelo guardado corrupto o incompatible.\nSe reentrenará.\nDetalles:\n{e}"
                    )
                    os.remove(self.modelo_path) # Borra modelo dañado
                    model = None # Forzar reentrenamiento


            #  Entrenar modelo si no existe o es corrupto 
            if model is None:
                model = self.crear_modelo_lstm((ventana, datos_escalados.shape[1]))  # Crea LSTM
                model.fit(X, y, epochs=200, verbose=0) # Entrena modelo
                model.save(self.modelo_path) # Guarda modelo
                with open(self.hash_path, "w") as f: # Guarda hash del archivo
                    f.write(hash_actual)
                messagebox.showinfo("Información", "Modelo entrenado y guardado correctamente.")

            # Predicción de las próximas 3 semanas
            secuencia_actual = df[['CLP', 'USD', 'EUR', 'Day', 'Month', 'Week']].values[-ventana:] # Últimas 12 semanas
            secuencia_escalada = scaler.transform(secuencia_actual).reshape(1, ventana, datos_escalados.shape[1]) # Escalar y dar forma
            pred_scaled = model.predict(secuencia_escalada) # Predicción escalada
            pred = scaler.inverse_transform(np.hstack([pred_scaled.reshape(3,3), np.zeros((3,3))]))[:, :3] # Desescalar predicción

            #Ultima fecha y prox. 3 semanas
            ultima_fecha = df['Date'].iloc[-1]
            fechas = [ultima_fecha + timedelta(weeks=i) for i in range(1, 4)]

            #  Mostrar resultados en la interfaz
            self.resultado = []
            self.resultado_texto.delete("1.0", END) # Limpia cuadro de texto
            for i, fecha in enumerate(fechas): # Recorre predicciones
                linea = f"Semana {i+1} ({fecha.date()}): CLP: {pred[i][0]:,.2f}, USD: {pred[i][1]:,.2f}, EUR: {pred[i][2]:,.2f}"
                self.resultado_texto.insert(END, linea + "\n")
                self.resultado.append({
                    'Semana': f"Semana {i+1}",
                    'Fecha': fecha.date(),
                    'CLP': pred[i][0],
                    'USD': pred[i][1],
                    'EUR': pred[i][2]
                })

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
        finally:
            self.label_procesando.config(text="")

    def guardar_excel(self):
        """Guardar resultados en un archivo Excel"""
        if not self.resultado:
            messagebox.showwarning("Advertencia", "No hay resultados para guardar.")
            return
        try:
            df_resultado = pd.DataFrame(self.resultado)
            archivo_guardar = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")]
            )
            if archivo_guardar:
                df_resultado.to_excel(archivo_guardar, index=False)
                messagebox.showinfo("Éxito", "Resultados guardados correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el archivo:\n{e}")

#%%

# Para ejecutar aplicación 
if __name__ == "__main__":
    root = Tk()
    app = PrediccionApp(root)
    
    
    Label(root, text="Autor: Fernanda Vilches C.", font=("Arial", 9, "italic"), fg="#993300").pack(side="bottom", pady=5)
    root.mainloop()
