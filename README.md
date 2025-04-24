# RP — Algorithmes CR & CNN

> _Projet « Voyageur canadien couvrant » — M2 RP 2024‑2025_
>
> Binôme : **Haya MAMLOUK – Doruk OZGENC**

Ce dépôt contient l’**implantation complète** des deux algorithmes :

- **CR — Cyclic Routing** (`cyclic_routing.py`)
- **CNN — Christofides‑Nearest‑Neighbour** (`cnn_routing.py`)

ainsi qu’un **harness** expérimental (`experiments.py`), un **notebook d’analyse** (`analysis.ipynb`) et le **rapport final**.

---

## 1 . Arborescence du projet

```
RP-OZGENC-MAMLOUK/
├── analysis.ipynb             # Notebook pour les graphs et les analyses
├── cyclic_routing.py          # Algorithme CR
├── cnn_routing.py             # Algorithme CNN
├── experiments.py             # Générateur et banc d’essai
├── results.csv                # Résultats numériques expérimentaux
├── requirements.txt           # Dépendances Python minimales
├── README.md                  # <– ce fichier
└── RP-OZGENC-MAMLOUK.pdf      # Rapport final au format PDF
```

> Le projet est conçu pour **Python ≥ 3.10**.

---

## 2 . Installation rapide

```bash
# 1‑ Créez un environnement dédié (recommandé)
python -m venv .venv
source .venv/bin/activate       # sous Windows : .venv\Scripts\activate

# 2‑ Installez les bibliothèques requises
pip install -r requirements.txt
```

Contenu de `requirements.txt` :

```
networkx>=3.0
numpy
matplotlib
pandas
tqdm
```

Aucune compilation nécessaire.

---

## 3 . Exécution des algorithmes seuls

### 3.1 Cyclic Routing (CR)

```bash
python cyclic_routing.py
```

_Lance un petit test sur un graphe complet ; imprime la tournée résultante._

Usage dans un code Python :

```python
from cyclic_routing import cyclic_routing
G = ...  # graphe complet avec poids
blocked = {(u, v)}
route, length = cyclic_routing(G, origin=0, blocked_edges=blocked)
```

### 3.2 Christofides‑Nearest‑Neighbour (CNN)

```bash
python cnn_routing.py
```

Ou via Python :

```python
from cnn_routing import cnn_routing
route, length = cnn_routing(G, origin=0, blocked_edges=blocked)
```

---

## 4 . Lancer les expériences

`experiments.py` compare systématiquement CR et CNN sur diverses familles de graphes et différentes valeurs de _k_ (nombre d’arêtes bloquées).

### 4.1 Familles d’instances testées

- **A** : petits graphes fixes
- **B** : points aléatoires uniformes
- **C** : clusters gaussiens
- **D** : grilles avec diagonales bon marché
- **E** : instances adversariales sur la tournée de Christofides

### 4.2 Exemple de commande complète

```bash
python experiments.py \
  --families A B C D E \
  --sizes 20 40 80 160 \
  --seeds 30 \
  --algos CR CNN \
  --timeout 30 \
  --out results.csv
```

Utilisez `tqdm` pour suivre la progression.

---

## 5 . Analyse et visualisation

Le fichier `analysis.ipynb` permet de générer tous les graphiques nécessaires pour comparer les performances des deux algorithmes selon :

- **la famille de graphes**,
- **le rapport de compétitivité**,
- **le temps d’exécution**,
- **la distribution des erreurs et cas extrêmes**,
- **l’impact du nombre d’arêtes bloquées (k)**.

```bash
jupyter lab  # puis ouvrir analysis.ipynb
```

---

## 6 . Résultats et conclusions

- CR (**Cyclic Routing**) est globalement **plus compétitif** mais plus lent.
- CNN (**Christofides-Nearest-Neighbor**) est **beaucoup plus rapide**, mais souvent un peu moins optimal.
- Famille D (grille avec diagonales) est l’un des seuls cas où CNN obtient des résultats proches ou légèrement supérieurs à CR.
- Famille E (adversariale) montre bien la robustesse de CR.

Voir le **rapport complet** dans `RP-OZGENC-MAMLOUK.pdf`.

---

## 7 . FAQ & dépannage

| Problème                                      | Solution                                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError: networkx`               | Exécuter `pip install -r requirements.txt`                                                       |
| Le benchmark semble figé                      | Utiliser `--sizes 20` et `--seeds 5` pour les tests rapides                                      |
| Certains CR échouent ou durent trop longtemps | Un timeout est géré automatiquement avec `--timeout`, les cas sont marqués `TIMEOUT` dans le CSV |

---

## 8 . Licence

Code publié sous licence **MIT** — libre de réutilisation et modification.

---

> © 2025 Doruk OZGENC & Haya MAMLOUK — Sorbonne Université\
> Master AI2D — Résolution de Problèmes
