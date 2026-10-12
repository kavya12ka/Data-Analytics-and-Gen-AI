# AI Fundamental Stock Analysis

A Jupyter/Colab notebook that pulls Microsoft (`MSFT`) financial data from Yahoo Finance, computes core fundamental and market-risk metrics, and then lets you **ask questions about the data in plain English** using a small Retrieval-Augmented Generation (RAG) pipeline built on Gemini embeddings and FAISS.

---

## What it does

1. **Collects data** with `yfinance`: annual income statement, balance sheet, cash flow, 5 years of daily prices, and company profile.
2. **Cleans and reshapes** each statement into one row per fiscal year (FY ends 30 June).
2. **Cleans and reshapes** each statement into one row per fiscal year (FY ends 30 June).
3. **Calculates fundamental metrics**
   - Growth: revenue, net income, diluted EPS
   - Profitability: gross / operating / net margin, ROE, ROA
   - Balance sheet health: debt-to-equity, current ratio, cash, net debt
   - Cash flow: operating cash flow, capex, free cash flow, FCF margin, capex as % of OCF, buybacks + dividends as % of FCF
4. **Calculates market metrics**: 52-week high/low, 1/3/5-year return, annualized return, annualized volatility, maximum drawdown, Sharpe ratio (0% risk-free rate).
5. **Builds a knowledge base**: turns each year's metrics into short text documents, embeds them with `gemini-embedding-001`, and stores them in a FAISS index.
6. **Answers questions**: retrieves the most relevant documents and sends them with your question to a Gemini model to generate a grounded answer.

## Results snapshot (Microsoft, FY2023 – FY2026)

| Metric | FY2023 | FY2024 | FY2025 | FY2026 |
|---|---|---|---|---|
| Revenue ($B) | 211.9 | 245.1 | 281.7 | 331.8 |
| Revenue growth | – | 15.7% | 14.9% | 17.8% |
| Operating margin | 41.8% | 44.6% | 45.6% | 46.8% |
| Net income ($B) | 72.4 | 88.1 | 101.8 | 133.7 |
| Net margin | 34.2% | 36.0% | 36.2% | 40.3% |
| Diluted EPS ($) | 9.68 | 11.80 | 13.64 | 17.95 |
| Operating cash flow ($B) | 87.6 | 118.5 | 136.2 | 182.9 |
| Capex ($B) | 28.1 | 44.5 | 64.6 | 115.9 |
| Free cash flow ($B) | 59.5 | 74.1 | 71.6 | 67.0 |
| Debt-to-equity | 0.29 | 0.25 | 0.18 | 0.13 |
| Current ratio | 1.77 | 1.27 | 1.35 | 1.23 |
| ROE | 35.1% | 32.8% | 29.7% | 30.2% |

**Price / risk (5 years to 8 Oct 2026):** 5-year return ≈ 85%, annualized return ≈ 13.1%, annualized volatility ≈ 28.2%, maximum drawdown ≈ −37.1%, Sharpe ≈ 0.46.

*Key takeaway from the data: revenue and profit keep growing, but capex has quadrupled in three years, so free cash flow has been falling since FY2024 even as operating cash flow rises.*

Figures are as of the date the notebook was run and will change when you re-run it.

## Tech stack

- **Python 3**, `pandas`, `numpy`
- **yfinance** – market and financial statement data
- **google-genai** – Gemini embeddings (`gemini-embedding-001`, 768 dimensions) and answer generation
- **FAISS** (`faiss-cpu`) – vector similarity search
- **LangChain text splitters** and **sentence-transformers** – imported for chunking/embedding experiments

## Getting started

### 1. Clone

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Run in Google Colab (recommended)

The notebook reads the API key from Colab Secrets, so Colab is the easiest route.

1. Open `AI_Fundamental_StockAnalysis.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
3. In Colab, open **🔑 Secrets**, add a secret named `GOOGLE_API_NAME` containing your key, and enable notebook access.
4. Run all cells.

### 3. Run locally

```bash
pip install yfinance pandas numpy faiss-cpu google-genai sentence-transformers langchain-text-splitters jupyter
```

`from google.colab import userdata` only works in Colab. Locally, replace the key lookup with an environment variable:

```python
import os
api_key = os.environ["GEMINI_API_KEY"]
```

Never commit your API key to the repository.

## Example questions

```python
ask("How has Microsoft's revenue and revenue growth changed over the years?")
ask("How much free cash flow does Microsoft generate and how much is returned to shareholders?")
ask("How has the stock performed historically and how risky is it?")
```

## Project structure

```
.
├── AI_Fundamental_StockAnalysis.ipynb   # Data collection, metrics, RAG pipeline
└── README.md
```

## Known limitations

- **Free-tier rate limits:** the Gemini free tier allows only a small number of generation requests per day; running all seven test questions in one go can return a `429 RESOURCE_EXHAUSTED` error.
- **Four fiscal years only:** Yahoo Finance returns about four years of annual statements, so the first year has no growth figures.
- **Simplified ratios:** ROE and ROA use year-end balances rather than averages; Sharpe ratio assumes a 0% risk-free rate; prices are dividend/split-adjusted, so returns include dividends.
- **Yahoo "Net Debt":** Yahoo's figure excludes lease obligations and short-term investments, so it is not equal to Total Debt − Cash and overstates Microsoft's net debt.
- **Single company, no valuation:** the notebook does not yet include valuation multiples (P/E, P/FCF) or peer comparison.

## Roadmap

- Put the fiscal year in every knowledge-base document so the retriever can tell years apart
- Normalize embeddings before using FAISS inner-product search
- Retrieve more chunks (or all years) for trend questions
- Add valuation metrics and multi-company comparison
- Add retry/back-off for API rate limits
- Make the ticker configurable

