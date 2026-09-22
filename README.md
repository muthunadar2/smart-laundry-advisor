# 🧺 AI-Based Smart Laundry Wash Setting Advisor

A college mini-project combining **LangChain + LLM** with a genuine **Fuzzy Inference System**.

## What the project does

The user describes laundry in normal language. LangChain sends the text to an LLM, which extracts:

- Load size
- Dirt level
- Water-saving priority
- Fabric type

The extracted numerical inputs are passed to a fuzzy inference system.

The fuzzy system performs:

1. Membership functions
2. Fuzzification
3. Rule evaluation
4. Aggregation
5. Centroid defuzzification

It then recommends Gentle, Normal, or Heavy washing.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set your API key before running:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="YOUR_KEY"

# macOS/Linux
export OPENAI_API_KEY="YOUR_KEY"
```

## Streamlit deployment

Upload this repository to GitHub.

On Streamlit Community Cloud, select `app.py` as the main file.

Add this secret:

```toml
OPENAI_API_KEY = "YOUR_KEY"
```

## Example input

"I have around 5 kg of cotton clothes. They are very dirty and I want to save water."

## Important

This is an academic decision-support demonstration, not a real washing-machine controller.
