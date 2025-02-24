import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Garantizar reproducibilidad
random_state = 42

def exploracion_datos(df):
    print("Información general del dataset:")
    print(df.info())
    print("\nDescripción estadística:")
    print(df.describe())
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    # Histograma de precios de las casas
    plt.figure(figsize=(8,5))
    plt.hist(df['SalePrice'], bins=30, edgecolor='black')
    plt.xlabel('Precio de Venta')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Precios de Casas')
    plt.show()

exploracion_datos(train_df)

# Análisis de agrupamiento: Vecindarios
agrupamiento = train_df.groupby('Neighborhood')['SalePrice'].mean().sort_values()
plt.figure(figsize=(12,5))
agrupamiento.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Vecindario')
plt.ylabel('Precio Promedio')
plt.title('Precio Promedio de Casas por Vecindario')
plt.xticks(rotation=90)
plt.show()

# Selección de variables numéricas relevantes
numericas = train_df.select_dtypes(include=['int64', 'float64']).dropna()
