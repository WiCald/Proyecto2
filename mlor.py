import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Descarga y carga de datos
# ===============================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
pd.set_option('display.float_format', '{:.3f}'.format)

# Garantizar reproducibilidad
random_state = 42

# ===============================
# 2. Análisis exploratorio extenso
# ===============================
def exploracion_datos(df):
    print("2. Análisis exploratorio extenso")
    print("Información general del dataset:")
    print(df.info())
    print("\nDescripción estadística:")
    print(df.describe())
    print("\nValores nulos por columna:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    
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

# ===============================
# 4. División del dataset (Entrenamiento y Prueba)
# ===============================
print("4. División del dataset (Entrenamiento y Prueba)")
train_df = train_df.sample(frac=1, random_state=random_state)  # Mezclar datos
train_size = int(len(train_df) * 0.8)
train_data = train_df[:train_size]
test_data = train_df[train_size:]

print(f"Datos de entrenamiento: {len(train_data)} filas")
print(f"Datos de prueba: {len(test_data)} filas")

# ===============================
# 5. Ingeniería de características
# ===============================
print("5. Ingeniería de características")
vars_seleccionadas = ['OverallQual', 'GrLivArea', 'TotalBsmtSF']
X_multi = (train_data[vars_seleccionadas] - train_data[vars_seleccionadas].median()) / train_data[vars_seleccionadas].std()
y_multi = train_data['SalePrice']

# ===============================
# 6. Regresión lineal univariada (GrLivArea vs SalePrice)
# ===============================
print("6. Regresión lineal univariada (GrLivArea vs SalePrice)")
X = train_data['GrLivArea']
Y = train_data['SalePrice']

# Cálculo de coeficientes de regresión
x_mean = X.mean()
y_mean = Y.mean()
b1 = sum((X - x_mean) * (Y - y_mean)) / sum((X - x_mean) ** 2)
b0 = y_mean - b1 * x_mean

print(f"Ecuación de regresión lineal: SalePrice = {b0:.2f} + {b1:.2f} * GrLivArea")

# Gráfica de regresión lineal
plt.figure(figsize=(8,5))
plt.scatter(X, Y, alpha=0.5, label='Datos reales')
plt.plot(X, b0 + b1 * X, color='red', label='Línea de regresión')
plt.xlabel('Área habitable sobre el suelo (GrLivArea)')
plt.ylabel('Precio de Venta')
plt.title('Regresión Lineal: Área Habitable vs Precio')
plt.legend()
plt.show()

# ===============================
# 7. Regresión lineal múltiple
# ===============================
print("7. Regresión lineal múltiple")
X_multi = pd.concat([pd.Series(1, index=X_multi.index, name='Intercept'), X_multi], axis=1)
coefs = pd.DataFrame(pd.DataFrame((X_multi.T @ X_multi)).pipe(lambda x: pd.DataFrame(np.linalg.inv(x.values), x.index, x.columns)) @ X_multi.T @ y_multi)

print("\nCoeficientes de la regresión lineal múltiple:")
print(coefs)

# ===============================
# 8. Evaluación del modelo en datos de prueba
# ===============================
print("8. Evaluación del modelo en datos de prueba")
X_test = (test_data[vars_seleccionadas] - train_data[vars_seleccionadas].median()) / train_data[vars_seleccionadas].std()
X_test = pd.concat([pd.Series(1, index=X_test.index, name='Intercept'), X_test], axis=1)
predicciones_test = X_test @ coefs

# Cálculo de métricas de error
mae = abs(test_data['SalePrice'] - predicciones_test.squeeze()).mean()
mse = ((test_data['SalePrice'] - predicciones_test.squeeze()) ** 2).mean()
print(f"\nMAE (Error Absoluto Medio): {mae:.2f}")
print(f"MSE (Error Cuadrático Medio): {mse:.2f}")

# ===============================
# 9. Comparación de Predicciones vs. Reales
# ===============================
print("9. Comparación de Predicciones vs. Reales(Ver grafica)")
plt.figure(figsize=(8,5))
plt.scatter(test_data['SalePrice'], predicciones_test, alpha=0.5)
plt.xlabel('Precio Real en Test')
plt.ylabel('Precio Predicho en Test')
plt.title('Evaluación del Modelo en Datos de Prueba')
plt.show()