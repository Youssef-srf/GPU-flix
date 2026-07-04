<div align="center">

# 🎬 GPU-Flix

**Accélérer un moteur de recommandation par filtrage collaboratif en le faisant passer du CPU au GPU.**

*Exercice technique 25 — Module Systèmes Répartis & Programmation Parallèle*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CuPy](https://img.shields.io/badge/CuPy-CUDA%2012-76B900?logo=nvidia&logoColor=white)](https://cupy.dev/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![Google Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#licence)

</div>

---

## 📌 Contexte

L'exercice 25 demande d'implémenter une multiplication de matrices sur GPU (CuPy ou CUDA) et de comparer son temps d'exécution à une implémentation CPU équivalente.

Plutôt qu'un micro-benchmark sur des matrices aléatoires, **GPU-Flix applique cet exercice à un vrai cas d'usage** : un système de recommandation par filtrage collaboratif sur le jeu de données **MovieLens 1M**, où la similarité entre chaque paire d'utilisateurs se calcule par un unique produit matriciel massif :

```
S = A · Aᵀ          (A : 6040 utilisateurs × 3706 films, normalisée L2)
```

Le CPU est limité par son nombre de cœurs physiques ; le GPU traite des milliers de produits scalaires simultanément via **cuBLAS**. Le gain croît avec le volume de calcul — exactement le profil de cette matrice de similarité.

---

## ⚡ Résultats

Benchmark académique, moyenne sur 3 exécutions, validation numérique systématique (`np.allclose`, `atol=1e-4`) :

| Taille matrice | CPU (NumPy) | GPU (CuPy) | Accélération | Validation |
|---|---|---|---|---|
| 1000 × 3706 | 0.0621 s | 0.0020 s | **×31.2** | ✅ CPU == GPU |
| 3000 × 3706 | 0.3649 s | 0.0160 s | **×22.9** | ✅ CPU == GPU |
| 6040 × 3706 (complet) | 1.5126 s | 0.0870 s | **×17.4** | ✅ CPU == GPU |

Sur l'application interactive (jeu complet 6040×6040, en conditions réelles) : **1.44 s (CPU) → 0.098 s (GPU), soit ×14.7**.

Chaque mesure GPU est encadrée par un **warm-up** (initialisation du contexte CUDA hors chronométrage) et une **synchronisation explicite** (`cp.cuda.Stream.null.synchronize()`), sans quoi seul le lancement asynchrone du kernel serait mesuré, pas le calcul réel.

---

## 🧩 Pipeline

```
Ingestion              Normalisation           Calcul dual              Validation
MovieLens 1M      →     Norme L2 / ligne   →    S = A · Aᵀ         →    np.allclose
6040 × 3706             (→ similarité             NumPy / CuPy            atol = 1e-4
(pivot)                  cosinus)                 en parallèle
```

La même matrice normalisée alimente les deux chemins de calcul (CPU et GPU) ; le résultat n'est accepté que si l'étape de validation confirme leur équivalence numérique.

---

## 🖥️ Démo — AI Streaming Engine

Une interface **Gradio** expose le moteur sous forme d'application type streaming :

- **6 040 profils réels** testables, chacun avec son propre historique de notes MovieLens.
- **Recommandations pondérées** : les films déjà vus sont exclus du score, les prédictions classées par pourcentage de correspondance.
- **Télémétrie en direct** : chaque clic relance la comparaison CPU vs GPU et affiche le graphique de temps correspondant.

---

## 🚀 Installation & exécution

Le projet est conçu pour tourner sur **Google Colab** avec un GPU (Tesla T4 ou équivalent) — aucune installation locale requise.

1. Ouvrez `Projet_GPU_Flix.ipynb` dans [Google Colab](https://colab.research.google.com/).
2. Activez le GPU : `Exécution` → `Modifier le type d'exécution` → `GPU (T4)`.
3. Exécutez les cellules dans l'ordre :

```bash
# Cellule 0 — Vérification GPU + installation CuPy
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip install -q cupy-cuda12x
```

```bash
# Cellule 1 — Benchmark académique + validation (Exercice 25)
!pip install -q gradio
!wget -q -nc https://files.grouplens.org/datasets/movielens/ml-1m.zip
!unzip -q -n ml-1m.zip
```

```bash
# Cellule 2 — Lance l'application Gradio interactive
# (affiche une URL publique du type https://xxxxx.gradio.live)
```

### Dépendances principales

| Bibliothèque | Rôle |
|---|---|
| `cupy-cuda12x` | Calcul matriciel sur GPU (équivalent NumPy, exécuté via CUDA/cuBLAS) |
| `numpy` / `pandas` | Calcul CPU de référence et manipulation des données |
| `gradio` | Interface web interactive de démonstration |
| `matplotlib` | Génération du graphique comparatif CPU vs GPU |

---

## 📂 Structure du dépôt

```
.
├── Projet_GPU_Flix.ipynb     # Notebook complet (benchmark + application Gradio)
├── TP_GPU_Flix_Enonce.docx        # TP pour les étudiants à faire
└── README.md
```

---

## 🔭 Perspectives

- **Passage à l'échelle** : valider la tendance sur MovieLens 25M.
- **Indexation approximative** : remplacer le calcul dense par une recherche ANN (ex. FAISS) au-delà d'un certain volume de catalogue.
- **Distribution multi-GPU** : répartir le calcul de similarité sur plusieurs cartes pour des catalogues de plusieurs millions d'utilisateurs.

---

## 👥 Auteurs

Projet réalisé par **Youssef Sarraf** et **Sifeddine Legnioui**
Encadré par Pr. Chaimaa Lamharmech, Pr. Rawane Azeroual et Pr. Yassir Ahrar

Master Intelligence Artificielle — Faculté des Sciences Ben M'Sik, Université Hassan II de Casablanca

## Licence

Ce projet est distribué à des fins pédagogiques dans le cadre du module Systèmes Répartis & Programmation Parallèle.
