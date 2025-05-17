import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

students_df = pd.read_csv("student-mat.csv", sep=";")

students_data = students_df[[ "studytime","G1","G2","failures","absences","G3"]]

x = students_data.drop("G3", axis = 1)
y = students_data["G3"]

x_training, x_test, y_training, y_test = train_test_split(x, y, test_size = 0.30, random_state=1)

model = LinearRegression()
model.fit(x_training, y_training)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro quadrático médio (MSE): {mse:.2f}")
print(f"Coeficiente de determinação (R²): {r2:.2f}")

plt.scatter(y_test, y_pred)
plt.xlabel("Notas Reais (G3)")
plt.ylabel("Notas Previstas")
plt.title("Comparação entre notas reais e previstas")
plt.plot([0, 20], [0, 20], 'r--')
plt.show()
