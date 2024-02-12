
# Import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Assign the filename: file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score,auc
import numpy as np

#dataset: https://www.kaggle.com/competitions/titanic
# Assign the filename: file
file = 'titanic.csv'

# Read the file into a DataFrame: df
df = pd.read_csv(file)

#df=df.dropna()
print("Valores Nulos detectados en df: ")
print(df.isna().sum())

print("columnas",df.columns)
sex_mapping = {'male': 0, 'female': 1}  # Codificación: male -> 0, female -> 1
df['SexNumer'] = df['Sex'].map(sex_mapping)
print(df)


print(df['Pclass'].values)

Supervivientes=df[df['Survived']==1]
print(Supervivientes)
print("Fallecidos",len(df[df['Survived']==0]))
print("Supervivientes",len(df[df['Survived']==1]))

sns.relplot(x='Fare', y="Age", kind="scatter", data=df, size="Fare", row="SexNumer", hue="Survived")
plt.show()

# Plot 'Class Died' variable in a histogram

sns.relplot(x='Fare', y="Age", kind="scatter", data=df, size="Fare", col="Pclass", hue="Survived")
plt.show()

# Filtrar los datos por clase para los que no sobrevivieron
ClassDied = df[df['Survived'] == 0]
# Contar la cantidad de pasajeros de cada clase que no sobrevivieron
died_counts = ClassDied['Pclass'].value_counts().sort_index()

# Filtrar los datos por clase para los que sobrevivieron
ClassNotDied = df[df['Survived'] == 1]
# Contar la cantidad de pasajeros de cada clase que sobrevivieron
not_died_counts = ClassNotDied['Pclass'].value_counts().sort_index()

# Crear un gráfico de barras para comparar las clases entre los que sobrevivieron y los que no
plt.bar(died_counts.index - 0.2, died_counts.values, width=0.4, align='center', label='Survived=0')
plt.bar(not_died_counts.index + 0.2, not_died_counts.values, width=0.4, align='center', label='Survived=1')

plt.xlabel('Pclass (1º, 2º, 3º class)')
plt.ylabel('Count')
plt.title('Distribution of Pclass for Survived=0 and Survived=1')
plt.xticks([1, 2, 3])  # Establecer las posiciones de las marcas del eje x
plt.legend()
plt.show()



#dfNumerico=df[['SexNumer', 'Survived','Age','Fare', 'Pclass','SibSp','Parch']]
# Calcula la matriz de correlación SexNumer y Survived

correlation_matrix = df[['SexNumer', 'Survived']].corr()

# Visualiza la matriz de correlación como un mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación entre SexNumer (0=Hombres, 1=Mujeres) y Survived (0=Fallecido, 1=Superviviente)')
plt.show()


# Calcula la matriz de correlación edad y superviencia
correlation_matrix = df[['Age', 'Survived']].corr()

# Visualiza la matriz de correlación como un mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación entre Edad y  Survived (0=Fallecido,1=Superviviente)')
plt.show()


# Calcula la matriz de correlación tarifa(fare)y superviencia
correlation_matrix = df[['Fare', 'Survived']].corr()
# Visualiza la matriz de correlación como un mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación Tarifa  y  Survived (0=Fallecido,1=Superviviente)')
plt.show()



# Calcula la matriz de correlación tarifa(fare)y superviencia
correlation_matrix = df[['Pclass', 'Survived']].corr()
# Visualiza la matriz de correlación como un mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación PcClass (1 = 1st, 2 = 2nd, 3 = 3rd) y  Survived (0=Fallecido,1=Superviviente)')
plt.show()



correlation_matrix = df[['SibSp', 'Survived']].corr()
# Visualiza la matriz de correlación como un mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación SibSp (of siblings / spouses aboard the) y  Survived (0=Fallecido,1=Superviviente)')
plt.show()




correlation_matrix = df[['Parch', 'Survived']].corr()
# Visualiza la matriz de correlación como un mapa de calor
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación Parch(of parents / children aboard thee) y  Survived (0=Fallecido,1=Superviviente)')
plt.show()

#parche R.L
dfNumerico=df[['SexNumer', 'Survived','Age','Fare', 'Pclass','SibSp','Parch']]
#dfNumerico=df[['SexNumer', 'Survived','Age','Fare', 'Pclass','SibSp','Parch']]
print("Valores Nulos detectados en dfNumerico ANTES**: ")
print(dfNumerico.isna().sum())
dfNumerico=dfNumerico.dropna()
print("Valores Nulos detectados en dfNumerico  DESPUÉS**: ")
print(dfNumerico.isna().sum())


# Paso 1: Dividir el DataFrame en características y la variable objetivo
#Modelo completo ROC:0.73:
#X = dfNumerico[['SexNumer','Age','Fare', 'Pclass','SibSp','Parch']]


#Modelo parcial roc 0.53
#X = dfNumerico[['SibSp','Parch']]


#Modelo parcial roc 0.73
#X = dfNumerico[['SexNumer', 'Fare', 'Pclass']]

#Modelo parcialmás parsimonioso roc 0.73
X = dfNumerico[['SexNumer', 'Fare']]

y = dfNumerico['Survived']

# Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 3: Inicializar el modelo de regresión logística
model = LogisticRegression()

# Paso 4: Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

#TOP 5 VARS

# Obtener los coeficientes del modelo
coeficientes = model.coef_[0]

# Crear un DataFrame para visualizar los coeficientes y las características correspondientes
coeficientes_df = pd.DataFrame({'Variable': X.columns, 'Coeficiente': coeficientes})
coeficientes_df['Coeficiente_Abs'] = np.abs(coeficientes_df['Coeficiente'])

# Ordenar los coeficientes por valor absoluto de mayor a menor
coeficientes_df = coeficientes_df.sort_values(by='Coeficiente_Abs', ascending=False)

# Imprimir las 5 variables que más peso predicen en Survived=1
print("Top 5 variables que más peso predicen en Survived=1:")
print(coeficientes_df.head(5))

# Paso 5: Evaluar el rendimiento del modelo utilizando los datos de prueba
y_pred = model.predict(X_test)

# Paso 6: Interpretar los resultados
accuracy = accuracy_score(y_test, y_pred)
print("Exactitud del modelo:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(conf_matrix)

report = classification_report(y_test, y_pred)
print("Informe de clasificación:")
print(report)

# Paso 7: Calcular la curva ROC y el área bajo la curva (AUC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Paso 8: Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

