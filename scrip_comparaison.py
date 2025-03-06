import matplotlib.pyplot as plt
import numpy as np

# Modèles et leurs scores d'accuracy
models = ["Decision Tree", "SVM", "Random Forest"]
accuracies = [0.9064, 0.7783, 0.9351]  # Accuracy des modèles

# Création du graphique
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['blue', 'orange', 'green'])

# Ajout des valeurs au-dessus des barres
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 0.005, f"{acc:.2%}", 
             ha='center', fontsize=12, fontweight='bold')

# Ajout d'une ligne de référence pour l'objectif de 80%
plt.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label="Seuil 80%")

# Personnalisation du graphique
plt.title("Comparaison des Modèles - Accuracy", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0.7, 1)  # Pour mieux voir les différences
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

# Affichage du graphique
plt.show()
