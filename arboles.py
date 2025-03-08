import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Cargar los datos
# ===============================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Seleccionar todas las variables predictoras y la variable objetivo
vars_seleccionadas = train_df.columns.drop(['SalePrice'])
X = train_df[vars_seleccionadas].select_dtypes(include=[np.number]).fillna(0).values
y = train_df['SalePrice'].values

# Dividir los datos en entrenamiento y prueba
def train_test_split_manual(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx].copy(), X[test_idx].copy(), y[train_idx].copy(), y[test_idx].copy()

X_train, X_test, y_train, y_test = train_test_split_manual(X, y)

# ===============================
# 2. Implementación de Árbol de Decisión para Regresión
# ===============================
class DecisionTreeRegressorManual:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y, depth=0):
        X, y = np.array(X), np.array(y)
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)
        
        best_split, best_mse = None, float('inf')
        for i in range(X.shape[1]):
            thresholds = np.unique(X[:, i])
            for t in thresholds:
                left_idx = np.where(X[:, i] <= t)[0]
                right_idx = np.where(X[:, i] > t)[0]
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                mse = (np.var(y[left_idx]) * len(left_idx) + np.var(y[right_idx]) * len(right_idx)) / len(y)
                if mse < best_mse:
                    best_split = (i, t)
                    best_mse = mse
        
        if not best_split:
            return np.mean(y)
        
        i, t = best_split
        left_idx, right_idx = np.where(X[:, i] <= t)[0], np.where(X[:, i] > t)[0]
        left_subtree = self.fit(X[left_idx].copy(), y[left_idx].copy(), depth + 1)
        right_subtree = self.fit(X[right_idx].copy(), y[right_idx].copy(), depth + 1)
        
        return (i, t, left_subtree, right_subtree)
    
    def predict_single(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        i, t, left, right = tree
        return self.predict_single(x, left if x[i] <= t else right)
    
    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])

# ===============================
# 3. Uso del Árbol de Decisión para Predecir y Analizar
# ===============================
regressor = DecisionTreeRegressorManual(max_depth=5)
regressor.tree = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mae = np.mean(np.abs(y_test - y_pred))
print(f"\nEvaluación del Árbol de Decisión para Regresión:")
print(f"MAE: {mae:.2f}")

# ===============================
# 4. Crear 3 Modelos Adicionales y Comparar
# ===============================
depths = [3, 7, 10]
for depth in depths:
    model = DecisionTreeRegressorManual(max_depth=depth)
    model.tree = model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)
    mae_model = np.mean(np.abs(y_test - y_pred_model))
    print(f"\nModelo con Profundidad {depth} - MAE: {mae_model:.2f}")

# ===============================
# 5. Comparación con Regresión Lineal
# ===============================
def regresion_lineal_manual(X, y):
    X = np.c_[np.ones(X.shape[0]), X]
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

beta = regresion_lineal_manual(X_train, y_train)
y_pred_lin = np.c_[np.ones(X_test.shape[0]), X_test] @ beta
mae_lin = np.mean(np.abs(y_test - y_pred_lin))
print(f"\nMAE del Árbol de Decisión: {mae:.2f}")
print(f"MAE de la Regresión Lineal: {mae_lin:.2f}")

# ===============================
# 6. Creación de Variable de Clasificación
# ===============================
def categorizar_precio(precio, limites):
    if precio < limites[0]:
        return 0  # Económicas
    elif precio < limites[1]:
        return 1  # Intermedias
    else:
        return 2  # Caras

limites_precio = [np.percentile(y, 33), np.percentile(y, 66)]
train_df['CategoriaPrecio'] = np.array([categorizar_precio(p, limites_precio) for p in train_df['SalePrice']])
y_class = train_df['CategoriaPrecio'].values

# ===============================
# 7. Construcción del Árbol de Clasificación y Visualización
# ===============================
class DecisionTreeClassifierManual(DecisionTreeRegressorManual):
    def predict(self, X):
        return np.array([round(self.predict_single(x, self.tree)) for x in X])

clf = DecisionTreeClassifierManual(max_depth=5)
clf.tree = clf.fit(X_train, y_class)
y_pred_clf = clf.predict(X_test)
accuracy = np.mean(y_pred_clf == y_class[:len(y_pred_clf)])
print(f"\nPrecisión del Árbol de Clasificación: {accuracy:.2f}")

plt.figure(figsize=(8, 5))
plt.hist(y_pred_clf, bins=3, alpha=0.7, label="Predicciones")
