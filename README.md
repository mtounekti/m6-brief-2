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
├── .env
├── docker-compose.yml
└── README.md
```

## Lancement

```bash
docker compose up --build
```

## Services

| Service | URL |
|---|---|
| FastAPI | http://localhost:8080 |
| Streamlit | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Prefect | http://localhost:4200 |

## Fonctionnement

L'utilisateur dessine un chiffre dans l'interface Streamlit. L'image est envoyée à FastAPI qui retourne une prédiction via le modèle CNN. Si la prédiction est incorrecte, l'utilisateur peut la corriger. La correction est stockée en base de données. Toutes les heures, Prefect analyse les corrections et déclenche un réentraînement si nécessaire.
