# Expériences « recherche hybride » — exécution sur le serveur

Dépôt **privé** : il contient des extraits de recommandations (droits des éditeurs) et les 100 questions cliniques,
dont la diffusion n'est pas encore décidée. Ne pas le rendre public tel quel.

| Étape | Ce qu'elle fait | Durée indicative (CPU, 8 cœurs) |
|---|---|---|
| `preflight` | vérifie Python, Docker, l'espace disque, le port, Ollama, l'accès Internet | < 1 min |
| `venv` | installe les dépendances (versions figées) dans `.venv/` | 3–5 min |
| `data` | télécharge SciFact, NFCorpus, TREC-COVID et FiQA (BEIR) | 5 min |
| `postgres` | lance un PostgreSQL 16 **séparé** (conteneur `hybrid-exp-pg`, port 127.0.0.1:18508) | 1 min |
| `bm25` | recherche BM25 + dense : 100 questions cliniques, TREC-COVID (171 000 documents), FiQA | 4–8 h |
| `fts` | recherche plein texte PostgreSQL (`ts_rank_cd`, étape ET puis OU) + règle A0, sur les 6 corpus | 1–2 h |
| `ollama` | lance un Ollama **dédié** (conteneur `hybrid-exp-ollama`, 8 CPU max, port 127.0.0.1:18509) et télécharge `qwen2.5:7b` (~5 Go) | 5–10 min |
| `llm` | contrôleurs LLM B1/B2, environ 830 questions × 4 appels | 5–18 h |
| `package` | crée `hybrid_results_<date>.tar.gz` à renvoyer | < 1 min |

Aucune donnée patient. Le registre en service n'est pas touché (conteneur et port à part, port lié à 127.0.0.1).

## Prérequis
- Python 3.12 ou 3.11, Docker, `git`, `curl`, `unzip` ; 15 Go libres ; accès Internet (corpus BEIR, modèle
  d'embeddings, image Docker `postgres:16`).
- Rien à installer pour le LLM : le script lance son propre Ollama dans un conteneur, limité à 8 processeurs pour ne
  pas pénaliser les autres services, et le supprime au nettoyage. Pour utiliser un Ollama existant à la place :
  `sudo OLLAMA_HOST_URL=http://adresse:port ./run_all.sh`. Les calculs d'embeddings sont eux aussi limités à 8
  processeurs (`THREADS`).

## Marche à suivre
Récupérer le dépôt (identifiants GitHub demandés, le dépôt étant privé) :
```bash
git clone https://github.com/hatembellaaj/hybrid-retrieval-exp.git ~/IA/hybrid-exp && cd ~/IA/hybrid-exp
```
```bash
./run_all.sh preflight
```
Si tout est OK, lancer en arrière-plan (le terminal peut être fermé) :
```bash
nohup ./run_all.sh > run.log 2>&1 &
```
Suivre :
```bash
tail -f run.log
```
Le script reprend où il s'est arrêté si on le relance (marqueurs dans `state/`). Si Docker exige `sudo`,
lancer `sudo -E ./run_all.sh ...`. Pour une mise à jour du code : `git pull`, puis relancer.

À la fin, renvoyer le fichier `hybrid_results_<date>.tar.gz` indiqué en dernière ligne de `run.log`.

## Nettoyage
```bash
sudo docker rm -f hybrid-exp-pg hybrid-exp-ollama && sudo docker volume rm hybrid-exp-ollama
```
```bash
rm -rf ~/IA/hybrid-exp
```

## Contenu
- `code/fts_pg.py` : recherche plein texte sur PostgreSQL (notre implémentation, section 3.1 de l'article).
- `code/fusion.py` : fusion RRF et convexe (notre implémentation, identique aux conventions de production).
- `code/prompts.py` : consignes figées des contrôleurs LLM (version `2026-11-16-v1`).
- `data/hemophilia-dev`, `data/hemophilia-clin` : corpus hémophilie et questions.
- Quand il n'y a pas de jugements (questions cliniques), les classements sont enregistrés sans être évalués : les jeux
  de test verrouillés sont évalués plus tard, une seule fois.
