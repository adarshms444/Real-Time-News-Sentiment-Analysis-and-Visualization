# Real-Time-News-Sentiment-Analysis-and-Visualization
This project implements an end-to-end **real-time data pipeline** that fetches live news headlines, performs sentiment analysis using a **PySpark ML model**, and visualizes the results on a **dynamic web dashboard**.

---

## Features
- **Real-Time Data Ingestion**: Fetches the latest business news headlines every 60 seconds from [NewsAPI](https://newsapi.org).
- **Machine Learning**: Uses a pre-trained Logistic Regression model with a PySpark ML pipeline to classify headlines as *positive*, *negative*, or *neutral*.
- **Stream Processing**: Spark Structured Streaming processes and analyzes incoming data in real time.
- **Interactive Dashboard**: A web dashboard built with **Dash + Plotly** shows:
  - KPI cards for quick insights  
  - Sentiment distribution pie chart  
  - Time-series sentiment trends  
  - Word clouds for positive & negative terms  
  - Interactive table of latest headlines  
- **Public Access**: Exposes the dashboard via **ngrok**, so you can share a temporary public URL.

---

## Project Architecture
The pipeline is designed in **three stages**:

1. **Data Collection**  
   - Fetches JSON news from NewsAPI every 60 seconds  
   - Stores raw text files locally  

2. **Real-Time Processing (Apache Spark)**  
   - Spark Structured Streaming monitors the input directory  
   - Applies the pre-trained ML model for sentiment classification  
   - Stores results in an in-memory table  

3. **Visualization & Delivery**  
   - Dash queries Spark results every 5 seconds  
   - Dashboard auto-updates with latest visualizations  
   - Ngrok exposes the server for sharing  

### 🔎 Architecture Diagram
![Project Architecture](f0ccda5a-304a-4c6e-8115-28b8f5e8427b.png)

---

## 🛠 Technologies Used
### Backend & Processing
- Python  
- Apache Spark (PySpark, MLlib, Structured Streaming)  
- Pandas  

### Frontend & Visualization
- Dash  
- Plotly  
- Dash Bootstrap Components  

### Services & APIs
- NewsAPI  
- Ngrok  

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.8+  
- Java 8+ (required for PySpark)  

### 2. Clone Repository
```bash
git clone <your-repository-url>
cd <repository-directory>
