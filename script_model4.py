import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Charger les données
data_path = 'final\\fusion_fichier.xlsx'  # Assure-toi que le chemin est correct
data = pd.read_excel(data_path)

# Supprimer les lignes où la partie gagnante est "inconnue"
print("Nombre d'observations avant suppression des 'inconnue':", len(data))
data = data[data['Partie_Gagnante_2017'] != 'inconnue']
print("Nombre d'observations après suppression des 'inconnue':", len(data))

# Sélection des features et de la cible
features = ['Gauche_2017', 'Milieu_2017', 'Droite_2017', 'Votes_Nuls_2017', 'Indicateur_Emploi_2017', 'Indicateur_Crime_2017']
target = 'Partie_Gagnante_2017'

# Afficher les informations sur la variable cible
print("\n=== Information sur la variable cible ===")
print(f"Nom de la variable cible: {target}")
print("\nDistribution des valeurs:")
print(data[target].value_counts())
print("\nNombre total d'observations:", len(data[target]))
print("\nValeurs uniques:", data[target].unique())

# Encodage de la variable cible si ce n'est pas numérique
le = LabelEncoder()
data[target] = le.fit_transform(data[target])

# Afficher la correspondance entre les classes encodées et les noms originaux
print("\nCorrespondance des classes encodées:")
for i, class_name in enumerate(le.classes_):
    print(f"{i}: {class_name}")

# Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Standardisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialisation du modèle de régression logistique avec plus d'itérations
logistic_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)

# Entraînement du modèle
logistic_model.fit(X_train_scaled, y_train)

# Prédiction sur l'ensemble de test
y_pred = logistic_model.predict(X_test_scaled)

# Calcul de l'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Obtenir les classes uniques réellement présentes dans les données
unique_classes = np.unique(y_test)
target_names = le.inverse_transform(unique_classes)

print("\nClasses uniques dans l'ensemble de test:", target_names)

# Affichage des résultats avec le bon nombre de classes
print("\nAccuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
