from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dane treningowe
samples = [
    [300, 10, 5],  # Kalorie, Białko (g), Błonnik (g)
    [500, 20, 8],
    [200, 5, 2],
    [700, 30, 10],
    [450, 35, 12],
    [1000, 4, 5],
]

# Oczekiwane wyniki (0 - Niezdrowe, 1 - Zdrowe)
targets = [0, 1, 0, 1, 1, 0]

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(samples, targets, test_size=0.25, random_state=42)

# Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tworzenie i trenowanie modelu sieci neuronowej
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Ewaluacja modelu na zbiorze testowym
y_pred = model.predict(X_test)

print("Dokładność modelu:", accuracy_score(y_test, y_pred))
print("\nMacierz konfuzji:\n", confusion_matrix(y_test, y_pred))
print("\nRaport klasyfikacyjny:\n", classification_report(y_test, y_pred))

# Przewidywanie klasy dla nowego posiłku
print("\nWprowadź dane dla nowego posiłku:")
calories = float(input("Kalorie: "))
protein = float(input("Białko (g): "))
fiber = float(input("Błonnik (g): "))

new_meal = [[calories, protein, fiber]]
new_meal_normalized = scaler.transform(new_meal)

prediction = model.predict(new_meal_normalized)

# Wyświetlanie wyniku
print(f"Nowy posiłek: {new_meal} => Klasyfikacja: {'Zdrowe' if prediction[0] == 1 else 'Niezdrowe'}")
