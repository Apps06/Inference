# Unbiased AI Decision

> A multi-agent AI debate system that audits datasets and models for hidden bias — built with LangGraph, xAI Grok, and Google Gemini.

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
# Edit .env and add your XAI_API_KEY and/or GOOGLE_API_KEY
```

### 3. Run the server
```bash
python -m backend.main
```

### 4. Open the UI
Visit: http://localhost:8000

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (cyclic StateGraph) |
| Models | xAI Grok 4.20 + Google Gemini 2.5 |
| Fairness Tools | Fairlearn |
| Backend | FastAPI + SSE streaming |
| Frontend | Vanilla HTML/CSS/JS (Perplexity + Grok UI style) |
| RL Logging | JSONL trajectories for future MARTI/PPO training |

## Model Support

**xAI Grok (via OpenAI-compatible API)**
- `grok-4.20-reasoning` (default, best quality)
- `grok-4.20-non-reasoning` (faster)
- `grok-4.1-fast-reasoning` (cost-efficient)

**Google Gemini (via google-genai SDK)**
- `gemini-2.5-pro`
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
