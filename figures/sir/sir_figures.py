import pickle
import lzma
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 
import sys
import types

os.chdir("..")
os.chdir("..")
sys.path.append(os.getcwd())
print(os.getcwd())
from utils_functions import Theta

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 0

# Construction d’un faux module utils contenant Theta
# (car le pickle cherche 'utils.Theta')
fake_utils = types.ModuleType("utils")
fake_utils.Theta = Theta
sys.modules["utils"] = fake_utils
with lzma.open('figures/sir/SIR_results{}.xz'.format(seed), 'rb') as f:
    data = pickle.load(f)
print("Data loaded from figures/sir/SIR_results{}.xz".format(seed), flush = True)
thetas_fr = data["Thetas_smc_fr"]
thetas_reg = data["Thetas_smc_reg"]
thetas_dep = data["Thetas_smc_dep"]
print("Thetas loaded", flush = True)
# Graphe $R_0$
f, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.kdeplot(thetas_dep.glob[:,0],  label='Departments scale', linewidth=2, linestyle='-', color='blue')
sns.kdeplot(thetas_reg.glob[:,0], label='Regions scale', linewidth=2, color='orange', linestyle='--')
sns.kdeplot(thetas_fr.glob[:,0], label='National scale', linewidth=2, color='green', linestyle='-.')

# Adding labels and title
plt.xlabel('$R_0$', fontsize=12)
plt.ylabel('Density', fontsize=12)


# Adding legend
plt.legend()

# Display the plo
plt.savefig("figures/sir/posterior_R0_{}.pdf".format(seed))
plt.close()
print("Plot posterior R0 saved in figures/sir/posterior_R0_{}.pdf".format(seed), flush = True)
# Graphe $\beta$

fig, ax = plt.subplots(1, 1, figsize=(15, 8))

j = 2
mus_fr = thetas_fr.loc[:,:,j]
mus_reg = thetas_reg.loc[:,:,j]
mus_dep = thetas_dep.loc[:,:,j]

sns.kdeplot(mus_fr[:,0], label='National scale', linewidth=2, linestyle='-.', color='green')
for k in range(mus_reg.shape[1]):
    sns.kdeplot(mus_reg[:,k], label='Regions scale', linewidth=2, color='orange', linestyle='--', alpha=0.5)

for k in range(mus_dep.shape[1]):
    sns.kdeplot(mus_dep[:,k], label='Departments scale', linewidth=2, color='green', linestyle='-.', alpha=0.5)
plt.xlabel('$\\delta$', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.savefig("figures/sir/posterior_delta_{}.pdf".format(seed))
plt.close()
# plt.legend()
print("Plot posterior delta saved in figures/sir/posterior_delta_{}.pdf".format(seed), flush = True)
# Cartes

import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Charger le fichier shapefile des départements
# (GeoDataFrame avec les formes des départements)
france = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson")

# 2. Exemple de dictionnaire : numéro du département → valeur mu
mu_dict = {"01": 0.1, "02": 0.2, "03": 0.3, "04": 0.4, "05": 0.5, "06": 0.6,
           "07": 0.7, "08": 0.8, "09": 0.9, "10": 1.0, "11": 1.1, "12": 1.2,
           "13": 1.3, "14": 1.4, "15": 1.5, "16": 1.6, "17": 1.7, "18": 1.8, 
           "19": 1.9,  "21": 2.2, "22": 2.3, "23": 2.4,
           "24": 2.5, "25": 2.6, "26": 2.7, "27": 2.8, "28": 2.9, "29": 3.0,
           "30": 3.1, "31": 3.2, "32": 3.3, "33": 3.4, "34": 3.5, "35": 3.6,
           "36": 3.7, "37": 3.8, "38": 3.9, "39": 4.0, "40": 4.1, "41": 4.2,
           "42": 4.3, "43": 4.4, "44": 4.5, "45": 4.6, "46": 4.7, "47": 4.8,
           "48": 4.9, "49": 5.0, "50": 5.1, "51": 5.2, "52": 5.3, "53": 5.4,
           "54": 5.5, "55": 5.6, "56": 5.7, "57": 5.8, "58": 5.9, "59": 6.0,
           "60": 6.1, "61": 6.2, "62": 6.3, "63": 6.4, "64": 6.5, "65": 6.6,
           "66": 6.7, "67": 6.8, "68": 6.9, "69": 7.0, "70": 7.1, "71": 7.2,
           "72": 7.3, "73": 7.4, "74": 7.5, "75": 7.6, "76": 7.7, "77": 7.8,
           "78": 7.9, "79": 8.0, "80": 8.1, "81": 8.2, "82": 8.3, "83": 8.4,
           "84": 8.5, "85": 8.6, "86": 8.7, "87": 8.8, "88": 8.9, "89": 9.0,
           "90": 9.1, "91": 9.2, "92": 9.3, "93": 9.4, "94": 9.5, "95": 9.6}

for i,key in enumerate(mu_dict.keys()):
    # Convertir les clés en chaînes de caractères
    mu_dict[key] = np.mean(thetas_dep.loc[:, i, j])
    
        
# 3. Ajouter les valeurs mu à la GeoDataFrame
france["$\\delta$"] = france["code"].map(mu_dict)

# 4. Affichage de la carte
fig, ax = plt.subplots(figsize=(8, 8))
france.plot(column="$\\delta$", ax=ax, cmap="viridis", edgecolor="black", legend = False)

# ax.set_title("Carte de France - Valeurs de μ par département", fontsize=14)
ax.axis("off")
plt.colorbar(ax.collections[0], ax=ax, fraction=0.03, pad=0.04)
plt.savefig("figures/sir/map_delta_{}.pdf".format(seed))
plt.close()
print("Map delta saved in figures/sir/map_delta_{}.pdf".format(seed), flush = True)