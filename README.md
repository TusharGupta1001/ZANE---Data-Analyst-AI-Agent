# 🌟 ZANE: Data Analyst Agent
**Insight at the Speed of Thought — No Code, Just Clarity**


---


## 🧠 What Does ZANE Stand For?

**ZANE** — *Zero-Shot Agent for Next-Gen Explanations*

| Letter | Meaning                                      | Description                                                                 |
|--------|----------------------------------------------|-----------------------------------------------------------------------------|
| **Z**  | **Zero-Shot**                                | No prior training needed — understands your data right away                 |
| **A**  | **Agent**                                    | Acts as an intelligent assistant to analyze, visualize, and explain         |
| **N**  | **Next-Gen**                                 | Uses modern AI (e.g., Together AI) and interactive tools like Plotly        |
| **E**  | **Explanations**                             | Delivers insights in plain English and visual form with AI-powered summaries|

✨ ZANE combines the **intelligence of an AI agent**, the **power of zero-shot learning**, and **creative explanations** to help you understand your data effortlessly — all without writing a single line of code.

⚡ Explore, visualize, and interpret data with the help of a zero-shot AI agent.


---


## 🌟 What Can ZANE Do?

ZANE empowers you to explore, analyze, and present your data effortlessly — no coding required.

- 📂 **Upload Files**  
  Supports CSV, Excel, TXT, DOCX, PDF, PNG, JPG  
  → Auto-cleans and preprocesses your data

- 📊 **Analyze Data**  
  Calculates means, medians, standard deviations, correlations  
  → Detects outliers and scores data quality (0–100)

- 📉 **Visualize Data**  
  Generates interactive charts using Plotly:  
  → Scatter, Histogram, Box, Line, and Heatmap plots

- ❓ **Ask Questions**  
  Use natural language queries like:  
  _“What’s the average Sales?”_ or _“Max of Annual_Fees?”_

- 📄 **Generate Reports**  
  Create and download custom PDF reports with selected insights and visuals

- 🤖 **AI Descriptions**  
  Get fun, creative data summaries like:  
  _“🎨 Sales shines at a mean of 1234.56!”_

- ⚖️ **Compare Groups**  
  Visualize differences between groups (e.g., Region vs. Sales) with box plots


---


### 🧠 Advanced Features

ZANE goes beyond basic analysis to offer powerful and intuitive tools for deeper insights.

- 📥 **Download Reports**  
  One-click PDF export powered by ReportLab

- 💡 **Insight Cards**  
  Highlights top insights with emojis (e.g., skewness, categories)  
  → Limited to 6 cards for clarity

- 🎯 **Feature Importance**  
  Uses `DecisionTreeClassifier` to rank key features impacting your target

- 🌟 **Data Quality Score**  
  Visual 0–100 score with emoji-based feedback  
  → 🥳 Excellent, 😐 Average, ⚠️ Needs Attention

- ⚖️ **Comparative Analysis**  
  Segment comparisons via box plots  
  → Choose columns like `Region` and compare metrics like `Sales`

- 🤖 **Smart Suggestions**  
  AI suggests follow-up questions based on your dataset  
  → e.g., “max Revenue”, “null value count”

- 📄 **Report Builder**  
  Choose what to include in your downloadable report:  
  → Summary, Insights, Visualizations, Comparisons

  
---


## 🛠️ Technologies Used in ZANE

| Category            | Technologies                            | Purpose                             |
|---------------------|------------------------------------------|-------------------------------------|
| Framework           | Streamlit 🖥️                             | Web app interface                   |
| Language            | Python 🐍                                 | Core logic                          |
| Data Handling       | Pandas, NumPy, SciPy, PyPDF2, docx2txt   | Data parsing and analysis           |
| Visualization       | Plotly 📉                                 | Interactive visualizations          |
| Machine Learning    | Scikit-learn (DecisionTreeClassifier)    | Feature importance                  |
| Styling             | Tailwind CSS 🎨                          | UI design                           |
| PDF Generation      | ReportLab 📄                              | Custom report generation            |
| AI Integration      | Together AI API 🤖                        | Creative AI dataset descriptions    |
| Utilities           | JSON, Base64, UUID, OS, IO               | Backend utility support             |


---


## ✨ Features at a Glance

### 📊 Core Functionalities

- **File Upload & Processing**
  - Supported: CSV, Excel, TXT, DOCX, PDF, PNG, JPG
  - Auto-handles missing values, dynamic data editing via interactive table

- **Statistical Analysis**
  - Means, medians, std dev, correlations, trend detection
  - Outlier detection via z-scores

- **Interactive Visualizations**
  - ✅ Scatter Plots  
  - 📊 Histograms  
  - 📦 Box Plots  
  - 📈 Line Charts  
  - 🔥 Heatmaps

- **Feature Importance**
  - Scikit-learn’s DecisionTreeClassifier
  - Bar chart for visual comparison

- **Data Quality Scoring**
  - Scores from 0–100 with emoji feedback 🥳 😐 ⚠️


---


### 📂 Upload Your Data

| ✅ Action              | 💬 What Happens                                                        |
|------------------------|------------------------------------------------------------------------|
| Upload CSV, Excel, TXT, PDF, DOCX, PNG, JPG | ZANE auto-parses your file, identifies missing values, and generates a **data quality score (0-100)** |
| Live Data Table        | Edit data on-the-fly in an interactive table                          |
| Smart Suggestions 🧹   | ZANE provides cleaning suggestions like "drop nulls" or "fill with median" |


---


### ❓ Ask Questions (Natural Language)

Use simple, natural language to explore your data. ZANE instantly interprets and responds with clear answers.

🗣️ **You ask:**

*"What's the mean of Annual_Fees?"*  
🤖 **ZANE replies:** `"The mean of Annual_Fees is ₹1234.56"`

🗣️ **You ask:**

*"Which Card_Category has the highest usage?"*  
🤖 **ZANE replies:** `"Business Card leads with 67 users"`

🗣️ **You ask:**

*"How many rows have null values?"*  
🤖 **ZANE replies:** `"23 rows contain missing data"`

✅ ZANE understands questions without needing specific commands — just ask like you're chatting with a data-savvy friend!


---


## 🎯 Use Cases

| 👤 User Type           | 🔧 Use Case                                                   |
|------------------------|---------------------------------------------------------------|
| 👩‍🎓 Students            | Submit clean data reports for academic projects               |
| 🧑‍💼 Business Analysts   | Explore KPIs without writing SQL or Python                    |
| 📈 Marketers           | Analyze campaign data, compare ROI by channel                 |
| 🧪 Data Scientists     | Quick EDA before modeling                                     |
| 🏢 Small Businesses    | Make informed decisions using your Excel/CSV files            |


---


## 🚀 Why ZANE Saves Time & Helps

ZANE is designed to make data analysis lightning-fast, intuitive, and enjoyable — no technical skills required.

- ⏱️ **No Coding Needed**  
  Upload → Click → Visualize — that’s it! No programming, no setup, just results.

- ⚡ **Fast Auto-Analysis**  
  Get statistics, charts, and insights in seconds — instantly after uploading your data.

- 🤓 **Smart Defaults**  
  Automatically selects the best visualizations and scoring methods to make analysis effortless.

- 🤖 **Built-in AI Assistance**  
  Supports natural language queries and generates fun, artistic summaries of your dataset.

- 📄 **Report Ready**  
  Download professional PDF reports with your analysis and visuals — ready to share!

✅ **ZANE streamlines the entire process** — from uploading raw files to getting presentation-ready insights — all in one click-friendly interface.


---


## 🖼️ Demo Video


---


## 📬 Contact & Feedback

Have questions, suggestions, or just want to say hi?
📧 Email: [tusharishangupta7@gmail.com](mailto:tusharishangupta7@gmail.com)


---
