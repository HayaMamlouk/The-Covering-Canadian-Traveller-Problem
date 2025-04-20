# RP — Algorithmes CR & CNN

> _Projet « Voyageur canadien couvrant » — M2 RP 2024‑2025_
>
> Binôme : **Haya MAMLOUK – Doruk OZGENC**\

Ce dépôt contient l’**implantation complète** des deux algorithmes :

- **CR — Cyclic Routing** (`cyclic_routing.py`)
- **CNN — Christofides‑Nearest‑Neighbour** (`cnn_routing.py`)

ainsi qu’un **harness** expérimental (`experiments.py`) et le rapport concernant le travail effectué.

---

## 1 . Arborescence du projet

```
RP-OZGENC-MAMLOUK/
├── analysis.ipynb             # Notebook pour les graphs et les analyses
├── cyclic_routing.py          # Algorithme CR
├── cnn_routing.py             # Algorithme CNN
├── experiments.py             # Générateur et banc d’essai
├── results.csv                # Resultâts numériques
├── requirements.txt           # Dépendances Python minimales
├── README.md                  # <– ce fichier
└── RP-OZGENC-MAMLOUK.pdf           # Rapport LaTeX final
```

> Le script sera testé sous **Python ≥ 3.10**.

---

## 2 . Installation rapide

```bash
# 1‑ Créez un environnement dédié (fortement recommandé)
python -m venv .venv
source .venv/bin/activate       # sous Windows : .venv\Scripts\activate

# 2‑ Installez les bibliothèques minimales
python -m pip install -r requirements.txt
```

`requirements.txt` contient :

```
networkx>=3.0
numpy     # utilisé indirectement
matplotlib
pandas
tqdm       # barre de progression dans experiments.py
```

Aucune compilation supplémentaire n’est nécessaire.

---

## 3 . Exécution des algorithmes seuls

### 3.1 Cyclic Routing (CR)

```bash
python cyclic_routing.py
```

_Lance le petit auto‑test interne ; imprime le tour et sa longueur._

Pour l’utiliser dans votre propre code :

```python
import networkx as nx
from cyclic_routing import cyclic_routing

G = nx.complete_graph(10)  # à vous de définir les poids
blocked = {(0, 3), (2, 5)}      # arêtes définitivement bloquées
route, length = cyclic_routing(G, origin=0, blocked_edges=blocked)
```

### 3.2 Christofides‑Nearest‑Neighbour (CNN)

Même principe :

```bash
python cnn_routing.py
```

ou :

```python
from cnn_routing import cnn_routing
route, length = cnn_routing(G, origin=0, blocked_edges=blocked)
```

---

## 4 . Lancer le banc d’essai complet

Le fichier `experiments.py` génère des graphes aléatoires, applique les deux algorithmes et enregistre les résultats **CSV** dans `results.csv`.

### 4.1 Options principales

| Option       | Valeur par défaut | Description                                                       |
| ------------ | ----------------- | ----------------------------------------------------------------- |
| `--families` | `B`               | Familles d’instances à tester (`A B C D E`)                       |
| `--sizes`    | `20 40 80 160`    | Tailles _n_ des graphes                                           |
| `--seeds`    | `30`              | Nombre de répétitions par (famille, n)                            |
| `--kvals`    | _calculé_         | Valeurs de _k_ (arêtes bloquées) ; si absent → `{0,⌊√n⌋,⌊0.3 n⌋}` |
| `--algos`    | `CR CNN`          | Algorithmes à évaluer                                             |
| `--out`      | `results.csv`     | Fichier de sortie                                                 |

### 4.2 Exemples

_Banc d’essai rapide pour la mise au point :_

```bash
python experiments.py --sizes 20 40 --seeds 5 --kvals 0 5
```

_Jeu complet (≈ 20 min sur un laptop) :_

```bash
python experiments.py                    # toutes les valeurs par défaut
```

Une barre de progression `tqdm` indique l’avancement.

---

## 5 . Génération des figures pour le rapport

Un notebook `analysis.ipynb` lit `results.csv`, calcule le rapport de compétitivité, trace les boîtes à moustaches et génère les graphiques utilisés dans `RP-nom1-nom2.pdf`.

```bash
jupyter lab
# puis ouvrez analysis.ipynb
```

---

## 6 . FAQ & dépannage

| Problème                                    | Solution                                                                                                          |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: networkx`             | Oubliez l’étape d’installation → `pip install -r requirements.txt`                                                |
| Le banc d’essai semble « bloqué »           | Utilisez `--sizes` plus petits ou `--seeds 5` pendant le débogage ; la barre `tqdm` montre la progression réelle. |
| CR lève _RuntimeError: no reachable vertex_ | Vérifiez que le générateur d’instances conserve la connectivité après suppression des arêtes bloquées.            |

---

## 7 . Licence

Code publié sous licence **MIT**. Vous êtes libres de le réutiliser en citant l’origine.

---

> © 2025 Doruk OZGENC & Haya MAMLOUK — Sorbonne Université\
> Master AI2D — Résolution de Problèmes
