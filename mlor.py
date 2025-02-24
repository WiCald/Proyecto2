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

# Análisis de correlación
correlacion = numericas.corr()['SalePrice'].sort_values(ascending=False)
print("\nCorrelaciones con el Precio de Venta:")
print(correlacion)

# Matriz de correlación para detectar multicolinealidad
plt.figure(figsize=(10,8))
plt.imshow(numericas.corr(), cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Matriz de Correlación')
plt.show()

# Visualización de la relación entre las variables más correlacionadas y el precio
plt.figure(figsize=(8,5))
plt.scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=0.5)
plt.xlabel('Área habitable sobre el suelo (GrLivArea)')
plt.ylabel('Precio de Venta')
plt.title('Relación entre Área Habitable y Precio de Venta')
plt.show()
