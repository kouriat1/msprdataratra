import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Chargement des données
# Chargement des données
data_path = 'final\\newfinalresultat.xlsx'
data = pd.read_excel(data_path)

# Sélection des features et de la target
features = ['Gauche_2017', 'Milieu_2017', 'Droite_2017', 'Votes_Nuls_2017', 'Indicateur_Emploi_2017', 'Indicateur_Crime_2017']
target = 'Partie_Gagnante_2017'

# Encodage de la variable cible
le = LabelEncoder()
data[target] = le.fit_transform(data[target])

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialisation du modèle SVC
svc_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Entraînement du modèle
svc_classifier.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = svc_classifier.predict(X_test)

# Calcul de l'accuracy
accuracy = accuracy_score(y_test, y_pred)

# Identification des classes uniques présentes
unique_classes = np.unique(np.concatenate((y_test, y_pred)))
unique_labels = le.inverse_transform(unique_classes)

# Génération du rapport de classification avec les noms de classes corrects
classification_report_result = classification_report(y_test, y_pred, target_names=unique_labels)

# Affichage des résultats
print("modèle de Support Vector Classifier\n")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)
