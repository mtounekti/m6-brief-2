# M5 Brief 3 – Monitoring complet d'une solution IA
### Prometheus + Grafana + Uptime Kuma + Discord

---

## Description

Mise en place d'une stack de monitoring complète pour superviser
la solution IA FastIA en production :
- **Prometheus** collecte les métriques de l'API et du système
- **Node Exporter** expose les métriques système (CPU, RAM, disk)
- **Grafana** visualise les métriques via le dashboard n°1860
- **Uptime Kuma** vérifie la disponibilité de l'API toutes les 60s
- **Discord Webhook** envoie des alertes en cas d'indisponibilité

---

## Structure du projet

```
fastia-m5-brief3-monitoring/
├── fastapi_app/
│   ├── main.py              # API FastAPI (routes /, /health, /data, /metrics)
│   ├── Dockerfile
│   └── requirements.txt
├── streamlit_app/
│   ├── app.py               # Frontend Streamlit
│   ├── Dockerfile
│   └── requirements.txt
├── prometheus/
│   └── prometheus.yml       # Config scraping (fastapi + node-exporter)
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboards.yml
│       └── datasources/
│           └── datasource.yml
├── .env                     # Variables d'environnement
├── docker-compose.yml       # Orchestration complète
└── README.md
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                        │
│                                                          │
│  ┌──────────┐    ┌────────────┐    ┌─────────────────┐  │
│  │ Streamlit│───▶│  FastAPI   │◀───│   Prometheus    │  │
│  │ :8501    │    │  :8080     │    │   :9090         │  │
│  └──────────┘    └────────────┘    └────────┬────────┘  │
│                                             │            │
│  ┌──────────┐    ┌────────────┐             │            │
│  │  Grafana │◀───│  Node      │◀────────────┘            │
│  │  :3000   │    │  Exporter  │                          │
│  └──────────┘    │  :9100     │                          │
│                  └────────────┘                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Uptime Kuma :3001  ──────▶  Discord Webhook     │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## Lancement

```bash
git clone https://github.com/mtounekti/fastia-m5-brief3-monitoring.git
cd fastia-m5-brief3-monitoring

# lancer tous les services
docker-compose up --build
```

---

## Services et ports

| Service | URL | Credentials |
|---|---|---|
| API FastAPI | http://localhost:8080 | — |
| Streamlit | http://localhost:8501 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Uptime Kuma | http://localhost:3001 | compte créé au setup |
| Node Exporter | http://localhost:9100 | — |

---

## Configuration

### `.env`

```bash
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
FASTAPI_PORT=8080
STREAMLIT_PORT=8501
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Prometheus (`prometheus/prometheus.yml`)

```yaml
scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['api:8080']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

---

## Grafana – Dashboard Node Exporter Full (n°1860)

### Import manuel

1. Connexion sur http://localhost:3000 (admin/admin)
2. **+** → **Import dashboard**
3. Entrer **`1860`** → **Load**
4. Sélectionner **Prometheus** comme datasource
5. **Import**

> Affiche : CPU usage, RAM, disk I/O, network — métriques système en temps réel

### Capture du dashboard
> *Ajouter ici une capture du dashboard Grafana 1860 avec les métriques système*

---

## Uptime Kuma – Monitoring disponibilité

### Configuration

- **Type** : HTTP(s)
- **URL surveillée** : `http://api:8080/health`
- **Intervalle** : 60 secondes
- **Notification** : Discord Webhook

### Capture Uptime Kuma
>  *capture*

---

## 🔔 Discord Webhook – Alertes automatiques

### Configuration dans Uptime Kuma

1. **Paramètres** → **Notifications** → **+ Ajouter**
2. **Type** : Discord
3. **Webhook URL** : `https://discord.com/api/webhooks/...`
4. **Tester** → message reçu ✅
5. **Enregistrer**

### Alertes reçues

| Événement | Message Discord |
|---|---|
| API **UP** | ✅ FastIA API est en ligne |
| API **DOWN** | 🔴 FastIA API est hors ligne |

### Capture Discord
> *Ajouter ici une capture des alertes Discord (UP/DOWN)*

---

## Tester les alertes

```bash
# simuler une panne
docker stop fastia-m5-brief3-monitoring-api-1

# Attendre ~60s => alerte Discord "API est DOWN"

# Relancer l'API – retour UP
docker start fastia-m5-brief3-monitoring-api-1

# Attendre ~60s => alerte Discord "API est UP"
```


---

*Brief M5 – Monitoring MLOps | FastIA 2025*