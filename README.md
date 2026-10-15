# 🖥️ PC Hardware Compatibility & Build Optimizer Agent

An autonomous AI agent built with **LangGraph**, **Pydantic v2**, and a **deterministic hardware compatibility engine** that translates natural language gaming and productivity requirements into fully compatible, budget-optimized PC builds.

---

## 🎯 The Engineering Problem

Large Language Models (LLMs) notoriously hallucinate when asked to configure complex PC hardware:
- Suggesting DDR5 memory on DDR4 motherboards.
- Pairing mismatched CPU sockets (e.g., AMD AM5 CPU with Intel LGA1700 motherboard).
- Underestimating power supply wattage requirements during transient GPU spikes.
- Ignoring physical clearance constraints (e.g., GPU length exceeding chassis internal volume).

### 💡 The Solution: Hybrid Neuro-Symbolic Agent
This project pairs **LLM semantic reasoning** (intent parsing and user explanation) with a **purely deterministic rule engine** (socket matching, clearance math, TDP headroom calculations) managed via a **LangGraph cyclic feedback loop**.

```
[User Request]
       │
       ▼
[1. Requirement Parser Node]
       │
       ▼
[2. Component Planner & Catalog Retriever Node] ◄────┐
       │                                              │ Self-Correction Loop
       ▼                                              │ (Max 3 retries)
[3. Deterministic Compatibility & Budget Validator] ──┘ (If conflicts detected)
       │ (When 100% compatible)
       ▼
[4. Synthesis & Performance Explainer Node]
       │
       ▼
[Final Recommendation & Game FPS Benchmarks]
```

---

## 🚀 Key Features

1. **Stateful Graph Architecture (LangGraph):**
   - Implements a cyclic graph with state checkpoints and error-driven self-correction.
2. **Zero-Hallucination Compatibility Engine:**
   - **Socket & Chipset:** Strict CPU-Motherboard socket pairing (AM4, AM5, LGA1700).
   - **Memory Architecture:** Matches DDR4/DDR5 standards across CPU memory controller and motherboard.
   - **Power Envelope:** Calculates peak system draw and enforces a minimum 25% PSU headroom buffer.
   - **Physical Clearances:** Validates GPU length vs. chassis clearance and CPU cooler height.
   - **Bottleneck Prevention:** Heuristic ratio guard ensuring balanced CPU/GPU performance tiers.
3. **Curated SQLite Hardware Catalog:**
   - Relational database storing verified modern hardware specs, TDP, benchmark scores, and market pricing in USD.
4. **Estimated In-Game Performance:**
   - Mathematical FPS estimation for popular titles (Cyberpunk 2077, Counter-Strike 2, The Witcher 3, Baldur's Gate 3) scaled to target resolution (1080p, 1440p, 4K).
5. **Observability Ready:**
   - Full tracing support via **LangSmith**.

---

## 🛠️ Tech Stack

- **Orchestration:** `LangGraph`, `LangChain Core`
- **Validation & Schemas:** `Pydantic v2`
- **Database:** `SQLite3`
- **Package & Environment Manager:** `uv`
- **Terminal UI:** `Rich`
- **Testing:** `Pytest`

---

## 📦 Project Structure

```text
├── src/
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine assembly and conditional routing
│   │   ├── nodes.py          # Execution nodes (Parse, Plan, Validate, Explain)
│   │   └── state.py          # Typed AgentState dictionary
│   ├── db/
│   │   ├── database.py       # SQLite connection and typed query layer
│   │   └── seed_data.py      # Hardware catalog (CPUs, GPUs, RAM, PSUs, Cases, etc.)
│   ├── models/
│   │   ├── build.py          # PCBuild and ValidationResult models
│   │   ├── hardware.py       # Pydantic schemas for hardware components
│   │   └── user_request.py   # Structured user requirements schema
│   ├── rules/
│   │   └── compatibility.py  # Pure deterministic compatibility rules
│   ├── config.py             # Environment configuration and paths
│   └── main.py               # Interactive CLI runner
├── tests/
│   ├── test_compatibility.py # Deterministic compatibility rule tests
│   └── test_database.py      # Database querying and seeding tests
├── pyproject.toml
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have Python 3.11+ installed. Using `uv` is recommended:
```powershell
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -e .
```

### 2. Configure Environment (Optional)
Copy the example environment file:
```powershell
cp .env.example .env
```
*(Note: An LLM API key is optional. When omitted, the agent automatically falls back to deterministic heuristic parsing, allowing fully offline testing).*

### 3. Run Automated Tests
```powershell
pytest -v
```

### 4. Launch the Interactive Agent CLI
```powershell
python -m src.main
```
Or pass a direct prompt:
```powershell
python -m src.main "Build a high-performance PC for Cyberpunk 2077 at 1440p, budget $1300"
```
