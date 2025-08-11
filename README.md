# 🧠 Human-Feedback-Safety-Simulator
 – AI Safety Research Platform

A showcase-ready platform for aligning LLMs using RLHF, harm taxonomies, and Chain-of-Thought reasoning. Built with Streamlit for researchers, engineers, and policy teams working on LLM safety.

---

## 🚀 Key Features

* Define and manage **harm taxonomies**
* Generate and annotate **LLM outputs**
* Train **reward models** via preference learning
* Evaluate safety using custom metrics
* ✅ Supports Chain-of-Thought Verifiers
* ✅ Works in **simulation mode** (no PyTorch required)
* 🧪 Research paper template + professional UI

---

## 🛠️ System Overview

### 🔧 Frontend

* `Streamlit` with multi-page layout
* Sidebar navigation, session state management

### 🔨 Backend Modules

* `TaxonomyManager`: harm categories
* `PromptGenerator`: create harmful/edge-case prompts
* `PreferenceTrainer`: pairwise ranking models
* `ChainOfThoughtVerifier`: CoT reasoning checks (🆕)
* `HarmEvaluator`, `ModelLoader`, `VisualizationManager`

---

## 📁 Data Structure

Stored in JSON for portability:

```text
📆 data/
 ├ taxonomy.json
 ├ prompts.json
 ├ raw_outputs.json
 ├ aligned_outputs.json
 ├ preferences.json
 ├ training_results.json
 ┗ verifications.json
```

Supports export in `.csv` or `.json`.

---

## 📉 Evaluation & Metrics

* Severity scores (1–5)
* Harm frequency comparison (pre/post alignment)
* CoT failure case tracking
* Real-time safe/unsafe reasoning detection

---

## 📦 Dependencies

| Area     | Libraries Used             |
| -------- | -------------------------- |
| LLM      | HuggingFace Transformers   |
| RLHF     | OpenRLHF / RLHFlow         |
| UI       | Streamlit, Plotly          |
| Data     | Pandas, NumPy, JSON        |
| Optional | CUDA, PyTorch, HF Datasets |

---

## 📦 Installation (Replit or Local)

```bash
git clone https://github.com/your-username/rlhf-studio.git
cd rlhf-studio
pip install -r requirements.txt
streamlit run app.py
```

> Works in Replit with GPU/CPU detection and simulation mode for no-hardware environments.

---

## 📊 Showcase Ready

* Live demo workflows
* Structured experiment directories
* Collaboration-ready layout
* ✅ Publication support template included

---

## 📚 Recent Updates (July 2025)

* ✅ CoT verifier integration
* ✅ Simulation mode (PyTorch-free)
* ✅ Professional UI for portfolios
* ✅ PyTorch-ready backend
* ✅ Policy alignment features added

---

## 🧪 Ideal Use Cases

* RLHF alignment experiments
* LLM safety evaluations
* AI policy interface prototyping
* Academic and applied AI research

---

## 🌐 License

MIT License

---

## 🙌 Contribute

Pull requests, ideas, and RLHF memes welcome.


The platform is designed as a self-contained research environment that can be easily deployed and used for AI safety research without requiring complex infrastructure setup.
