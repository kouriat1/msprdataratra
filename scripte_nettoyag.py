import pandas as pd

# Chargement du fichier Excel
file_path = "base\\resultat20221.xlsx"
df = pd.read_excel(file_path)

# Filtrer les lignes avec le code du département 92
df_92 = df[df['Code du département'] == "92"]

# Affichage du DataFrame filtré
print(df_92.head())

# Sauvegarder le fichier filtré
df_92.to_csv("resultats_dept_92.csv", index=False, sep=";")
