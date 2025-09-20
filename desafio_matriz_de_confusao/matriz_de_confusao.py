import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

matriz_confusao = {
    'VP': 100,
    'VN': 50,
    'FP': 10,
    'FN': 5
}
def calcular_acuracia(matriz_confusao):
    vp = matriz_confusao['VP']
    vn = matriz_confusao['VN']
    fp = matriz_confusao['FP']
    fn = matriz_confusao['FN']
    return (vp + vn) / (vp + vn + fp + fn)

def calcular_sensibilidade(matriz_confusao):
    vp = matriz_confusao['VP']
    fn = matriz_confusao['FN']
    if (vp + fn) == 0:
        return 0
    return vp / (vp + fn)

def calcular_especificidade(matriz_confusao):
    vn = matriz_confusao['VN']
    fp = matriz_confusao['FP']
    if (vn + fp) == 0:
        return 0
    return vn / (vn + fp)

def calcular_precisao(matriz_confusao):
    vp = matriz_confusao['VP']
    fp = matriz_confusao['FP']
    if (vp + fp) == 0:
        return 0
    return vp / (vp + fp)

def calcular_fscore(matriz_confusao):
    precisao = calcular_precisao(matriz_confusao)
    sensibilidade = calcular_sensibilidade(matriz_confusao)
    if (precisao + sensibilidade) == 0:
        return 0
    return 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

acuracia = calcular_acuracia(matriz_confusao)
sensibilidade = calcular_sensibilidade(matriz_confusao)
especificidade = calcular_especificidade(matriz_confusao)
precisao = calcular_precisao(matriz_confusao)
fscore = calcular_fscore(matriz_confusao)

print(f"Acurácia: {acuracia}")
print(f"Sensibilidade: {sensibilidade}")
print(f"Especificidade: {especificidade}")
print(f"Precisão: {precisao}")
print(f"F-score: {fscore}")

conf_matrix_array = np.array([
    [matriz_confusao['VN'], matriz_confusao['FP']],
    [matriz_confusao['FN'], matriz_confusao['VP']]
])

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_array, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Previsto Negativo', 'Previsto Positivo'],
            yticklabels=['Real Negativo', 'Real Positivo'])

plt.title('Matriz de Confusão')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.show()