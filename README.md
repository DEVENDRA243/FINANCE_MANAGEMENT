# FinGenius - AI Finance Advisor 💰

FinGenius is an AI-powered financial management system that helps you track, categorize, and analyze your personal finances using Google's Gemini AI.

## 🚀 Features

- **AI Categorization**: Automatically categorizes your transactions using Gemini 1.5/2.0/2.5 Flash.
- **Interactive Dashboards**: Visualize your spending patterns with Plotly charts.
- **Budget Allocation**: Smart budget suggestions based on your spending.
- **PDF Reports**: Generate detailed financial reports in PDF format.
- **Chat Interface**: Ask questions about your finances using a LangGraph-powered AI agent.

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **AI/ML**: Google Gemini API, LangGraph, LangChain
- **Database**: ChromaDB (Vector Store for chat context)
- **Data Visualization**: Plotly, Pandas

## 📋 Prerequisites

- Python 3.10+
- Google Gemini API Key

## ⚙️ Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd AI_FINACE_SYSTEM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🐳 Docker Support

You can also run the app using Docker:
```bash
docker build -t fingenius .
docker run -p 8501:8501 fingenius
```

## 🌐 Deployment

### 1. Streamlit Cloud (Recommended)
The easiest way to deploy this app is using [Streamlit Cloud](https://share.streamlit.io/).
1. Push this code to your GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io/) and click **"New app"**.
3. Select the repository `FINANCE_MANAGEMENT` and the `app.py` file.
4. Add your `GOOGLE_API_KEY` in the **Secrets** section.

### 2. Render
You can also deploy it on [Render](https://render.com/) using the provided `Dockerfile`.
1. Create a new **Web Service** on Render.
2. Connect your GitHub repository.
3. Render will automatically detect the `Dockerfile` and deploy the app.
4. Add `GOOGLE_API_KEY` to the **Environment Variables**.

---
Created with ❤️ by [Your Name/GitHub Handle]
