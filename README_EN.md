# SwingIQ ⚾
> AI-Powered Baseball Swing Analysis Service

🔗 **[Live Demo](https://swingiq.streamlit.app)**

Most existing swing analysis apps provide only surface-level feedback like "insufficient weight transfer." SwingIQ goes further — explaining **why** the issue occurs and **how** to fix it, with feedback personalized to each individual player.

Designed by a developer who played elite baseball from elementary school through high school, SwingIQ combines a custom biomechanics-based analysis engine with Claude AI to deliver coach-level feedback.

The long-term goal is to strengthen competitive baseball in Korea — from amateur players to national team athletes.

## Vision
- Personalized AI coaching for amateur and recreational players
- Linking KBSA performance data with swing mechanics
- Combining embedded sensor data with video analysis
- Expansion to all positions and a national-team-level data platform

## System Architecture
Video Upload → MediaPipe (joint/pose extraction) → utils.py (computes 7 biomechanical metrics) → Claude API (AI feedback generation) → Streamlit (web UI)

## Analysis Metrics
- Hip/shoulder rotation angle
- Kinetic Chain Gap (upper/lower body separation timing)
- Head stability
- Trail elbow position
- Trail knee angle
- Z-axis-based hip rotation (depth estimation)
- Automatic left-/right-handed batter branching

## Tech Stack
- Python 3.10
- MediaPipe 0.10.21
- OpenCV
- Claude API (Anthropic)
- Streamlit
- scikit-learn (Random Forest)
- joblib

## AI Model
- Random Forest Classifier (94.6% cross-validation accuracy)
- Training data: 39 swings from professional and amateur players
- Key features: head stability, shoulder rotation, elbow distance, Kinetic Chain Gap, wrist position
- Combines a swing score (0–100) with Claude AI-generated feedback

## How to Run
```bash
conda activate baseball-analyzer
streamlit run app.py
```

## Environment Setup
Create a `.streamlit/secrets.toml` file and add your API key:
```toml
ANTHROPIC_API_KEY = "your-api-key"
```

## Development Period
February 2026 – Present

## Roadmap
- Phase 1: Enhanced swing analysis + KBSA performance data integration
- Phase 2: InBody (body composition) data integration, swing history tracking
- Phase 3: Mobile app, back-view analysis, migration to FastAPI
- Phase 4: Spring Boot integration, embedded sensor integration
