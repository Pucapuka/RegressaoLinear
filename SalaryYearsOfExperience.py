#bibliotecas que poderemos usar
#Subindo os dados
from google.colab import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/Pucapuka/RegressaoLinear/main/Salary_Data.csv"
dataset = pd.read_csv(url)

#Análise Exploratória dos Dados
print("\n=== Informações Gerais ===")
print(dataset.info())  # Tipos de dados e não-nulos

print("\n=== Primeiras Linhas ===")
print(dataset.head())

print("\n=== Estatísticas Descritivas ===")
print(dataset.describe())

print("\n=== Valores Únicos ===")
print("YearsExperience:", dataset['YearsExperience'].nunique(), "valores únicos")
print("Salary:", dataset['Salary'].nunique(), "valores únicos")

# Verificação de Valores Ausentes
print("\n=== Valores Ausentes ===")
print(dataset.isnull().sum())

# Análise de Outliers com Boxplot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
dataset.boxplot(column=['YearsExperience'])
plt.title('Boxplot - Anos de Experiência')

plt.subplot(1, 2, 2)
dataset.boxplot(column=['Salary'])
plt.title('Boxplot - Salário')
plt.show()

#Histogramas para Distribuição
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
dataset['YearsExperience'].hist(bins=15)
plt.title('Distribuição - Anos de Experiência')

plt.subplot(1, 2, 2)
dataset['Salary'].hist(bins=15)
plt.title('Distribuição - Salário')
plt.show()

# Scatter Plot da Relação
plt.scatter(dataset['YearsExperience'], dataset['Salary'])
plt.title('Relação entre Experiência e Salário')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()

#TRATANDO OS DADOS
def tratar_dado_nao_numerico(df, estrategia='media', limite_remocao=0.3):
    """
    Trata valores NaN substituindo por média/mediana ou removendo colunas/linhas
    
    Parâmetros:
    df: DataFrame de entrada
    estrategia: 'media', 'mediana' ou 'remover'
    limite_remocao: % máximo de NaN para remover coluna/linha (0 a 1)
    """
    df_tratado = df.copy()
    
    # Remover colunas com muitos NaN
    colunas_remover = [col for col in df.columns if df[col].isnull().mean() > limite_remocao]
    df_tratado.drop(colunas_remover, axis=1, inplace=True)
    
    # Tratar valores restantes
    for col in df_tratado.columns:
        if df_tratado[col].isnull().any():
            if estrategia == 'media':
                valor = df_tratado[col].mean()
            elif estrategia == 'mediana':
                valor = df_tratado[col].median()
            elif estrategia == 'remover':
                df_tratado = df_tratado.dropna(subset=[col])
                continue
                
            df_tratado[col].fillna(valor, inplace=True)
    
    return df_tratado

def tratar_outliers(df, coluna, metodo='iqr', fator=1.5):
    """
    Identifica e trata outliers por IQR ou Z-score
    
    Parâmetros:
    coluna: Nome da coluna numérica
    metodo: 'iqr' (default) ou 'zscore'
    fator: Multiplicador do limiar (1.5 para IQR, 3 para Z-score)
    """
    serie = df[coluna].copy()
    
    if metodo == 'iqr':
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        limiar_inf = Q1 - fator * IQR
        limiar_sup = Q3 + fator * IQR
    else:  # zscore
        zscore = (serie - serie.mean()) / serie.std()
        limiar_inf = -fator
        limiar_sup = fator
    
    # Substituir outliers pela mediana
    outliers = (serie < limiar_inf) | (serie > limiar_sup)
    serie[outliers] = serie.median()
    
    return serie

#tratando o dado subido
df_tratado = tratar_dado_nao_numerico(dataset, estrategia='media')
df_tratado['YearsExperience'] = tratar_outliers(df_tratado, 'YearsExperience', metodo='iqr')
df_tratado['Salary'] = tratar_outliers(df_tratado, 'Salary', metodo='iqr')

X = df_tratado['YearsExperience'].values.reshape(-1, 1)
y = df_tratado['Salary'].values


#Normalização Z-score
def normalizar(x):
  return (x - np.mean(x)) / np.std(x)

X_norm = normalizar(X)
y_norm = normalizar(y)

#Divisão de dados em Treino/Teste
def dividir_dados(X, y, tamanho_teste=0.3):
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  divisao = int(len(X) * (1 - tamanho_teste))
  
  X_train = X[indices[:divisao]]
  X_test = X[indices[divisao:]]
  y_train = y[indices[:divisao]]
  y_test = y[indices[divisao:]]
  
  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = dividir_dados(X_norm, y_norm)


# Verificação das dimensões
print(f"Dimensão de X_train: {X_train.shape}")
print(f"Dimensão de y_train: {y_train.shape}")
print(f"Dimensão de X_test: {X_test.shape}")
print(f"Dimensão de y_test: {y_test.shape}")

#Classe RegressaoLinearManual:

class RegressaoLinearManual:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.historico_perdas = []

    def fit(self, X, y):
        n_amostras = X.shape[0]
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
          y_pred = X @ self.weights + self.bias
          dw = (2 / n_amostras) * X.T @ (y_pred - y)
          db = (2 / n_amostras) * np.sum(y_pred - y)
          self.weights -= self.lr * dw.reshape(self.weights.shape)
          self.bias -= self.lr * db
            
          self.historico_perdas.append(np.mean((y_pred - y)**2))

    def predict(self, X):
        return X @ self.weights + self.bias

# Treinamento
model = RegressaoLinearManual(lr=0.1, n_iters=500)
model.fit(X_train, y_train)

#Plot da Convergência
plt.plot(model.historico_perdas)
plt.title('Convergência do Gradiente Descendente')
plt.xlabel('Iterações')
plt.ylabel('MSE')
plt.show()

#Métricas de avaliação
y_pred = model.predict(X_test)

def calcular_metricas(y_true, y_pred):
  mse = np.mean((y_true - y_pred)**2)
  rmse = np.sqrt(mse)
  mae = np.mean(np.abs(y_true - y_pred))

  #R²
  ss_total = np.sum((y_true - np.mean(y_true))**2)
  ss_res = np.sum((y_true - y_pred)**2)
  r2 = 1 - (ss_res / ss_total)

  return mse, rmse, mae, r2

mse, rmse, mae, r2 = calcular_metricas(y_test, y_pred)

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R²: {r2:.4f}')

#Visualização dos resultados

plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', linewidth = 2, label='Regressão') # Corrigido linewideth para linewidth
plt.title('Predição de salário (Normalizado)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.legend()
plt.show()
