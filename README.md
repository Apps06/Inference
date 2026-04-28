# Unbiased AI Decision

> A multi-agent AI debate system that audits datasets and models for hidden bias — built with LangGraph and Google Gemini.

## How it works

1. You upload a CSV dataset (or just describe a scenario in the chat)
2. A team of 6 specialized AI agents runs in parallel and debates across 2 rounds:
   - **Data Statistician** — runs fairness metrics
   - **Fairness Auditor** — applies formal fairness definitions (disparate impact, etc.)
   - **Domain Expert** — adds real-world context for the use case
   - **Bias Adversary** — steelmans counter-arguments
   - **Ethical & Legal Reviewer** — checks DPDP Act, GDPR, EEOC compliance
   - **Final Judge** — synthesizes everything into a bias score + report
3. Every debate is logged as a JSONL trajectory for future MARTI RL training

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the server
```bash
python -m backend.main
```

### 4. Open the UI
Visit: http://localhost:8000

## Google Cloud Deployment

This prototype is ready for deployment to **Google Cloud Run**.

### 1. Build & Push to Artifact Registry
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/inference
```

### 2. Deploy to Cloud Run
```bash
gcloud run deploy inference \
  --image gcr.io/YOUR_PROJECT_ID/inference \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="GOOGLE_API_KEY=your_key"
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (cyclic StateGraph) |
| Models | Google Gemini 2.5 |
| Fairness Tools | Fairlearn |
| Backend | FastAPI + SSE streaming |
| Frontend | Vanilla HTML/CSS/JS |
| RL Logging | JSONL trajectories for future MARTI/PPO training |

## Model Support

**Google Gemini (via google-genai SDK)**
- `gemini-2.5-pro` (default, best quality)
- `gemini-2.5-flash`
- `gemini-3-flash-preview`

## RL Trajectory Logging

Every completed debate is saved to `backend/rl/trajectories/debates.jsonl`.  
Each record includes:
- Full agent transcript (all rounds)
- Fairness metrics (ground truth)
- Final report
- Multi-component reward score for training

To use with MARTI for RL training:
```bash
git clone https://github.com/TsinghuaC3I/MARTI.git
# Follow MARTI setup instructions, then point to debates.jsonl
```

## Sample Dataset

`sample_data/hiring_bias.csv` — a synthetic hiring dataset with intentional gender and caste bias for demonstration purposes.
