# 🤝 Autonomous Negotiation Agent

An AI-powered negotiation application that simulates interactions between buyer and seller agents to reach mutually acceptable prices. The application combines machine learning predictions with configurable negotiation strategies to make the negotiation process smarter and more interactive.

## 🌐 Live Demo

**[🚀 Try the Autonomous Negotiation Agent](https://negotiation-agent.streamlit.app/)**

## 📌 Project Overview

The Autonomous Negotiation Agent is a Streamlit-based application designed to simulate real-world buyer-seller negotiations.

Users can configure buyer and seller personalities, set a maximum buyer budget, select a product, and define the maximum number of negotiation rounds. The system uses machine learning insights to estimate negotiation success and expected final prices.

The application also provides analytics and negotiation history to help users understand negotiation outcomes.

## ✨ Key Features

- **🤝 Interactive Negotiation Simulator:** Simulates negotiations between autonomous buyer and seller agents.
- **🛒 Product Selection:** Choose products and view their market prices, categories, and seller minimum prices.
- **👤 Configurable Agent Personalities:** Customize buyer and seller personalities, such as balanced.
- **💰 Budget Management:** Set the buyer's maximum purchasing budget.
- **🔄 Negotiation Rounds:** Configure the maximum number of negotiation rounds.
- **🧠 Machine Learning Insights:** Predict negotiation success probability, failure probability, and expected final price.
- **📊 Analytics Dashboard:** Explore negotiation-related analytics.
- **📜 Negotiation History:** Review previously recorded negotiations.
- **🗄️ Database Management:** Store and manage negotiation records using a database.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning |
| Joblib | Loading saved ML models |
| Matplotlib | Data visualization |
| SQLite | Database management |

## 📂 Project Structure

```text
negotiation-agent/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── negotiation_model.pkl
│   ├── price_model.pkl
│   └── products.csv
│
├── database/
│   └── negotiation.db
│
├── src/
│   ├── agents.py
│   ├── db.py
│   ├── ml_model.py
│   ├── simulator.py
│   └── utils.py
│
└── .gitignore
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

**1. Clone the repository**

```bash
git clone https://github.com/NT-0017/negotiation-agent.git
```

**2. Navigate to the project directory**

```bash
cd negotiation-agent
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the virtual environment**

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the application**

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🖥️ Application Preview

The application provides an interactive dashboard with:

- Negotiation Simulator
- Machine Learning Insights
- Analytics
- Negotiation History
- Database Management

## 🔗 Project Links

- **Live Application:** https://negotiation-agent.streamlit.app/
- **GitHub Repository:** https://github.com/NT-0017/negotiation-agent

## 🎯 Project Objective

The main objective of this project is to demonstrate how autonomous agents and machine learning can be combined to simulate intelligent buyer-seller negotiations, estimate negotiation outcomes, and support better pricing decisions.

---

⭐ If you find this project interesting, feel free to explore the repository and try the live application!
