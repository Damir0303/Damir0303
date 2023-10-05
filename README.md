from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Загружаем набор данных Iris
iris = load_iris()
X = iris.data  # Матрица признаков
y = iris.target  # Вектор меток классов

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создаем модель метрического классификатора (K-Nearest Neighbors)
k = 3  # Количество ближайших соседей (можно выбрать другое значение)
model = KNeighborsClassifier(n_neighbors=k)

# Обучаем модель на обучающей выборке
model.fit(X_train, y_train)

# Делаем прогнозы на тестовой выборке
y_pred = model.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')
