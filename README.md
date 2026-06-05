# M6 - Brief 2 - Test, feedback et réentraînement dynamique sur MNIST

Application complète de classification de chiffres manuscrits avec boucle de feedback utilisateur.

## Structure du projet

```
m6-brief-2/
├── fastapi_app/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── streamlit_app/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       └── datasources/
├── models/
│   └── cnn_mnist.keras
├── data/
├── flow.py
├── prefect.yaml
├── Dockerfile
├── requirements.txt
├── .env
├── docker-compose.yml
└── README.md
```

## Architecture de la boucle de feedback

```
┌─────────────┐     image      ┌─────────────┐
│  Streamlit  │ ────────────▶  │   FastAPI   │
│  (canvas)   │ ◀────────────  │  /predict   │
└─────────────┘   prédiction   └─────────────┘
       │                              │
  correction                    stockage
   utilisateur                  SQLite
       │                              │
       ▼                              ▼
┌─────────────┐              ┌─────────────────┐
│   FastAPI   │              │     data/       │
│  /correct   │ ──────────▶  │ corrections.db  │
└─────────────┘              └─────────────────┘
                                      │
                               toutes les 15 min
                                      │
                                      ▼
                             ┌─────────────────┐
                             │     Prefect     │
                             │  analyse seuil  │
                             └─────────────────┘
                                      │
                              si >= 5 corrections
                                      │
                                      ▼
                             ┌─────────────────┐
                             │    FastAPI      │
                             │    /retrain     │
                             │  + Optuna       │
                             └─────────────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │  cnn_mnist      │
                             │  .keras mis     │
                             │  à jour         │
                             └─────────────────┘
```

## Lancement

```bash
docker compose up --build
```

## Déploiement du flow Prefect

Après le lancement des services, créer le work pool et déployer le flow :

```bash
docker exec -it prefect-worker prefect work-pool create mnist-pool --type process
docker exec -it prefect-worker prefect deploy --all
```

Le flow s'exécute toutes les 15 minutes et déclenche un réentraînement si le seuil de corrections est atteint.

## Services

| Service | URL |
|---|---|
| FastAPI | http://localhost:8080 |
| Streamlit | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Prefect | http://localhost:4200 |
| Uptime Kuma | http://localhost:3001 |

## Fonctionnement

L'utilisateur dessine un chiffre dans l'interface Streamlit. L'image est envoyée à FastAPI qui retourne une prédiction via le modèle CNN. Si la prédiction est incorrecte, l'utilisateur peut la corriger avec l'image associée. La correction est stockée en base SQLite.

Toutes les 15 minutes, Prefect analyse les nouvelles corrections. Si le seuil de 5 corrections est atteint, un réentraînement est déclenché automatiquement via la route `/retrain`. Le modèle est optimisé avec Optuna et sauvegardé.

## Variables d'environnement

| Variable | Valeur par défaut | Description |
|---|---|---|
| RETRAIN_THRESHOLD | 5 | Nombre de corrections avant réentraînement |
| CORRECTION_REPEAT_FACTOR | 20 | Surpondération des corrections face à MNIST |
| DB_PATH | data/corrections.db | Chemin vers la base SQLite |
| MODEL_PATH | models/cnn_mnist.keras | Chemin vers le modèle CNN |