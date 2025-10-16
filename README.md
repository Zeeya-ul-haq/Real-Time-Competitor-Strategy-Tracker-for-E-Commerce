🛍️ E-Commerce Competitor Analysis & Price Monitoring System
🚀 A full-stack AI-driven solution for scraping, analyzing, and visualizing e-commerce data from Amazon and Flipkart.
📚 Table of Contents

Overview

Project Structure

Modules & Files Explained

How It Works

Setup Instructions

Environment Variables (.env)

How to Run

Alerts & Notifications

Tech Stack

Future Enhancements

License

🧭 Overview

This project automates the scraping, cleaning, analysis, and visualization of mobile product data from Amazon and Flipkart.
It includes:

Web scrapers for both platforms

Automated data cleaning & ML model training

Streamlit dashboard for competitor analysis

Email-based price drop alerts

🗂️ Project Structure
AI-Batch/
│
├── amazon_scraper.py
├── flipkart_scraper.py
├── data_processor.py
├── ingestion.py
├── ml_trainer.py
├── config_manager.py
├── main.py
├── dashboard.py
├── config.yaml
│
├── alerts/
│   └── alerts.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md

🧩 Modules & Files Explained

Below is a detailed description of all 11 Python files and their purposes:

1️⃣ main.py

Purpose:
Acts as the orchestrator — it runs all modules in sequence:

Amazon Scraper

Flipkart Scraper

Data Cleaning & Processing

ML Model Training

Dashboard Launch

Key Functions:

Interactive CLI to choose which step to execute

Logs results and errors

Handles exceptions gracefully

Run Command:

python main.py

2️⃣ amazon_scraper.py

Purpose:
Scrapes product and review data from Amazon India.

Features:

Uses Selenium + BeautifulSoup for dynamic scraping

Extracts product title, price, ratings, and reviews

Saves data into scraped_data/amazon_products.csv

Enhancement Tip:
Add more categories (e.g., laptops, TVs) by editing the search URL in config.yaml.

3️⃣ flipkart_scraper.py

Purpose:
Scrapes product and review data from Flipkart India.

Features:

Handles pagination and multiple listing pages

Extracts product info, pricing, and reviews

Saves CSVs to My_docs/mobile_products.csv and My_docs/product_reviews.csv

Extra:
Supports price drop detection and email alerts.

4️⃣ data_processor.py

Purpose:
Cleans and validates scraped data for ML & analytics.

Steps:

Removes duplicates and outliers

Fills missing values

Normalizes numeric columns

Exports cleaned_mobile.csv and cleaned_reviews.csv

Output Folder:
processed_data/

5️⃣ ml_trainer.py

Purpose:
Trains machine learning models for price prediction.

Models Used:

Linear Regression

Random Forest

XGBoost

Outputs:

Trained model files in /models

Model performance metrics (accuracy, MAE, RMSE)

Command:

python ml_trainer.py

6️⃣ dashboard.py

Purpose:
Interactive Streamlit-based dashboard to visualize analytics.

Features:

Price trends

Competitor comparison

Sentiment analysis

Model performance section

Run Command:

streamlit run dashboard.py

7️⃣ config_manager.py

Purpose:
Loads and manages settings from config.yaml.

Features:

Centralized configuration handling

Validates YAML structure

Provides safe access to scraper, ML, and dashboard configs

8️⃣ config.yaml

Purpose:
Stores all configurable project settings:

URLs

Max pages

File paths

Dashboard options

You can tweak scraping depth, file outputs, and UI themes from here.

9️⃣ ingestion.py

Purpose:
Handles the loading, merging, and validation of multiple data sources.
Ensures that Amazon & Flipkart data can be combined safely.

Example:
Merges both platform data into a unified dataset for model training.

🔟 alerts/alerts.py

Purpose:
Detects price drops and sends alerts via email.

Functions:

detect_price_drops(csv_path, threshold)

send_email_alert(alerts_df, recipient)

Email Setup:
Uses credentials stored in .env and Gmail’s SMTP.

1️⃣1️⃣ My_docs/ Folder

Purpose:
Stores raw scraped data before cleaning.
Files include:

mobile_products.csv

product_reviews.csv

🧠 How It Works

1️⃣ Scrapers collect product data →
2️⃣ Data Processor cleans and validates →
3️⃣ ML Trainer builds price prediction models →
4️⃣ Dashboard visualizes results →
5️⃣ Alert system monitors price drops

⚙️ Setup Instructions
# 1. Clone the repository
git clone https://github.com/yourusername/ecommerce-competitor-analysis.git
cd ecommerce-competitor-analysis

# 2. Create virtual environment
python -m venv myenv
myenv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your .env file (for Gmail alerts)
# 5. Run main orchestrator
python main.py

🔐 Environment Variables (.env)
EMAIL_USER=youremail@gmail.com
EMAIL_PASS=your-app-password
HEADLESS=True
MAX_PAGES=3

🛎️ Alerts & Notifications

When a price drop is detected, an automatic email notification is sent.

from alerts.alerts import detect_price_drops, send_email_alert

alerts = detect_price_drops("My_docs/mobile_products.csv")
if not alerts.empty:
    send_email_alert(alerts, "yourmail@gmail.com")

🧰 Tech Stack
Category	Tools Used
Language	Python 3.10+
Web Scraping	Selenium, BeautifulSoup
Data Processing	Pandas, NumPy
ML	Scikit-learn, XGBoost
Dashboard	Streamlit
Config	YAML, dotenv
Notifications	SMTP (Gmail)
🚀 Future Enhancements

Add Telegram/Slack alerts

Expand to other categories (TVs, Laptops)

Cloud deployment (AWS Lambda / Streamlit Cloud)

API integration for live updates

🪪 License

MIT License © 2025 — Zeeya ul haq
