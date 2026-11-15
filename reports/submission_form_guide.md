# 📝 Official Submission Form Guide & Pre-Filled Answers
## ALGIONS AI LABS PRIVATE LIMITED — Advanced Project Assessment

**Internship Role:** AI & Machine Learning  
**Assigned Project:** Cinema Ticket Demand Forecasting  
**Official Reference:** `05_Cinema_Ticket_Demand_Forecasting.docx`  

---

### Section 1: Candidate Identification
| Field | Value / Guidance |
|---|---|
| **Full Name** | *[Enter your full legal name]* |
| **Registration Number** | *[Enter your university / candidate registration number]* |
| **University Email** | *[Enter your registered university / institutional email]* |

---

### Section 2: Assignment Confirmation
| Field | Value |
|---|---|
| **Internship Role** | `AI & Machine Learning` |
| **Exact Assigned Project Title** | `Cinema Ticket Demand Forecasting` |
| **Domain** | `Cinema Hall Ticket Booking` |

---

### Section 3: Code and Version
| Field | Value |
|---|---|
| **Repository URL** | `https://github.com/navneet210605/Cinema-Ticket-Demand-Forecasting` |
| **Primary Branch** | `main` |
| **Final Commit ID** | *[Use latest Git commit hash, e.g. `0eccbb0` or output of `git rev-parse HEAD`]* |
| **Repository Access** | Public (or provide read access to Algions AI Labs reviewer email if set to private) |

---

### Section 4: Demonstration & Live Deployment
| Field | Value |
|---|---|
| **Live Web Application URL** | `https://splendorous-meerkat-ef61cf.netlify.app/` |
| **Local API Endpoint** | `http://127.0.0.1:8000` (FastAPI/Starlette ASGI) |
| **Local Streamlit URL** | `http://localhost:8501` |
| **3–5 Minute Video Link** | *[Paste your Loom / YouTube unlisted / Google Drive video link]* |
| **Video Presentation Coverage** | Problem framing ($T-6\text{h}$ cutoff), live inference with 100-seat auditorium visualizer, 4-model leaderboard, cold-start Day 0 evaluation, feature importance, and test suite. |

---

### Section 5: Documentation & Artifact Locations
| Field | Value |
|---|---|
| **README Link** | [README.md](https://github.com/navneet210605/Cinema-Ticket-Demand-Forecasting/blob/main/README.md) |
| **Technical Report Link** | [reports/technical_report.md](https://github.com/navneet210605/Cinema-Ticket-Demand-Forecasting/blob/main/reports/technical_report.md) |
| **Model Evaluation Report** | [reports/model_evaluation_report.md](https://github.com/navneet210605/Cinema-Ticket-Demand-Forecasting/blob/main/reports/model_evaluation_report.md) |
| **Dataset Documentation** | [reports/dataset_documentation.md](https://github.com/navneet210605/Cinema-Ticket-Demand-Forecasting/blob/main/reports/dataset_documentation.md) |
| **Architecture Diagram** | [reports/architecture_diagram.md](https://github.com/navneet210605/Cinema-Ticket-Demand-Forecasting/blob/main/reports/architecture_diagram.md) |
| **Dedicated EDA Notebook** | [notebooks/01_Exploratory_Data_Analysis.ipynb](https://github.com/navneet210605/Cinema-Ticket-Demand-Forecasting/blob/main/notebooks/01_Exploratory_Data_Analysis.ipynb) |
| **End-to-End Notebook** | [notebooks/01_EDA_and_Modeling.ipynb](https://github.com/navneet210605/Cinema-Ticket-Demand-Forecasting/blob/main/notebooks/01_EDA_and_Modeling.ipynb) |
| **Serialized Model Artifact** | `models/best_model.joblib` (Champion Gradient Boosting Regressor Pipeline) |

---

### Section 6: Technology Stack & Known Limitations
| Field | Value |
|---|---|
| **Programming Language** | Python 3.13 |
| **Core Libraries** | Scikit-Learn 1.4+, Pandas 2.2+, NumPy 1.26+, SciPy 1.12+, Joblib 1.3+ |
| **Web & API Frameworks** | Starlette / FastAPI, Uvicorn, Streamlit 1.32+, HTML5, CSS3, Vanilla JS, Chart.js |
| **Visualization & EDA** | Matplotlib 3.8+, Seaborn 0.13+, Plotly 5.18+ |
| **Testing** | Pytest 8.0+ (19 automated unit tests passing at 100%) |
| **Known Assumptions & Scope Limitations** | 1. Features are captured strictly at $T-6\text{h}$; walk-in volume over final 6 hours constitutes predicted residual.<br>2. Extreme unmodeled events (blizzards, power outages, local emergencies) require human managerial override.<br>3. Predictions provide advisory decision support for staffing and screen sizing, not automated employment or firing decisions. |

---

### Section 7: Role-Specific Evaluation Metrics & Results
| Metric / Item | Value on Held-Out Chronological Test Set (375 Shows) |
|---|---|
| **Champion Model** | **Gradient Boosting Regressor** (160 trees, learning rate 0.07, max depth 5) |
| **Test WAPE (%)** | **8.87%** (vs 23.43% Capacity Baseline, 62.1% error reduction) |
| **Test MAE** | **11.06 tickets** (average prediction error across screens up to 450 seats) |
| **Test RMSE** | **15.20 tickets** |
| **Test R² Score** | **0.9476** (94.8% of variance explained) |
| **Validation WAPE (%)** | 8.13% |
| **Cold-Start Cohort (≤ 3d)** | **5.77% WAPE**, 9.18 MAE, 0.9562 R² ($N=26$ shows) |
| **Day 0 Premieres** | **6.89% WAPE**, 12.40 MAE, 0.9072 R² ($N=7$ shows) |
| **Dataset Origin** | Physics-informed synthetic simulation (`seed=42`, 2,500 records), 100% reproducible via `src/data_generation.py` |

---

### Section 8: AI & Attribution
| Field | Value |
|---|---|
| **AI Tools Used** | Antigravity AI Assistant (Google DeepMind) |
| **Purpose of Use** | Architecture auditing, Pytest test case construction, LaTeX math formatting, and pre-rendering high-resolution diagnostic charts. |
| **Data & Library Attribution** | Scikit-Learn, Pandas, Plotly, Starlette, Chart.js. Theatrical decay modeled after empirical box-office literature. |

---

### Section 9: Candidate Declarations
- [x] **Originality:** I certify that this project represents my own technical engineering work and understanding.
- [x] **Explainability:** I am fully prepared to explain all architectural decisions, mathematical formulas, and code during the 15-minute technical walkthrough.
- [x] **Live Debugging:** I am prepared to execute and debug code modifications during the 10-minute live coding assessment.
- [x] **Data Privacy & Secrets:** No real customer PII, secret keys, or `.env` files with credentials are committed to the repository (only `.env.example`).
- [x] **Link Verification:** All repository, documentation, Netlify, and notebook links are verified and publicly accessible.
