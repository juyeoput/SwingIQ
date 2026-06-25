[한국어](README_KO.md) | English

# SwingIQ ⚾
> AI-Powered Baseball Swing Analysis Service

🔗 **[Live Demo](https://swingiq.streamlit.app)**

Most existing swing analysis apps provide only surface-level feedback like "insufficient weight transfer." SwingIQ goes further — explaining **why** the issue occurs and **how** to fix it, with feedback personalized to each individual player.

Designed by a developer who played elite baseball from elementary school through high school, SwingIQ combines a custom biomechanics-based analysis engine with Claude AI to deliver coach-level feedback.

## Vision
- Personalized AI coaching for amateur and recreational players
- Linking individual performance data with swing mechanics
- Combining embedded sensor data with video analysis
- Expansion to all positions and a national-team-level data platform

## Screenshots

### Player Info & Video Upload
<img width="806" height="843" alt="Screenshot 2026-06-26 at 1 34 17 AM" src="https://github.com/user-attachments/assets/16de2e6f-2a63-401f-9d50-82a485dbeef1" />
<img width="806" height="843" alt="Screenshot 2026-06-26 at 1 37 27 AM" src="https://github.com/user-attachments/assets/bd3ae116-685a-4b14-beba-aaf4fc6797df" />

### Swing Analysis Results
<img width="675" height="887" alt="Screenshot 2026-06-26 at 1 38 39 AM" src="https://github.com/user-attachments/assets/edc45a72-1c41-4035-ae17-20d122026c34" />

### AI Coaching Feedback
<img width="680" height="816" alt="Screenshot 2026-06-26 at 1 39 25 AM" src="https://github.com/user-attachments/assets/36ba0512-f29c-47fe-8465-9b3882a08993" />

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

## Project Structure
```
SwingIQ/
├── app.py                  # Main app (Streamlit)
├── utils.py                 # Core analysis logic (pose extraction, metric computation)
├── requirements.txt
├── packages.txt
├── models/
│   └── swing_model.pkl      # Trained Random Forest model
├── data/
│   └── swing_dataset.csv    # Training dataset
└── dev/                      # Development scripts (not used in deployment)
    ├── extract_data.py       # Extract dataset from videos
    ├── pose_test.py          # Real-time pose analysis debugging
    ├── feedback.py           # AI feedback prompt testing
    └── train_model.ipynb     # Model training notebook
```

## How to Run

1. Clone this repository and navigate into the project folder.
2. (Optional, but recommended) Create a virtual environment:
```bash
conda create -n baseball-analyzer python=3.10
conda activate baseball-analyzer
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
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
