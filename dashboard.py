"""
Enhanced E-Commerce Competitor Strategy Dashboard
Now includes ML price predictions and is fully configurable
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pickle
import yaml
import logging
from pathlib import Path

# ML imports
try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Scikit-learn not available. Install with: pip install scikit-learn")

# Sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Email functionality  
try:
    from dotenv import load_dotenv
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

# Load configuration
@st.cache_data
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("config.yaml not found!")
        return {}

# ---------------- Streamlit Page Config ----------------
config = load_config()
dashboard_config = config.get('dashboard', {})

st.set_page_config(
    page_title=dashboard_config.get('title', 'E-Commerce Dashboard'),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {font-size:2.3rem;color:#1f77b4;text-align:center;margin-bottom:1rem;}
.section-header {font-size:1.6rem;color:#2e86ab;margin-top:2rem;margin-bottom:1rem;}
.positive-sentiment { color:#28a745; }
.negative-sentiment { color:#dc3545; }
.neutral-sentiment  { color:#ffc107; }
.metric-card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #007bff;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- ML Price Predictor ----------------
class MLPricePredictor:
    def __init__(self):
        self.models = {}
        self.preprocessing_objects = {}
        self.load_models()

    def load_models(self):
        """Load trained ML models and preprocessing objects"""
        try:
            model_dir = Path("models")

            # Load preprocessing objects
            preprocessing_path = model_dir / "preprocessing_objects.pkl"
            if preprocessing_path.exists():
                with open(preprocessing_path, 'rb') as f:
                    self.preprocessing_objects = pickle.load(f)

            # Load models
            model_files = {
                "Linear Regression": "linear_regression_model.pkl",
                "Random Forest": "random_forest_model.pkl", 
                "Gradient Boosting": "gradient_boosting_model.pkl",
                "XGBoost": "xgboost_model.pkl"
            }

            for model_name, filename in model_files.items():
                model_path = model_dir / filename
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)

        except Exception as e:
            st.warning(f"Could not load ML models: {e}")

    def predict_price(self, mobile_name, source, rating=4.0, discount=10.0):
        """Predict price for a mobile with given features"""
        try:
            if not self.models or not self.preprocessing_objects:
                return None

            # Extract brand from mobile name
            brand = mobile_name.split()[0] if mobile_name else "Unknown"

            # Prepare feature vector
            features = {}

            # Encode categorical features
            if 'label_encoders' in self.preprocessing_objects:
                encoders = self.preprocessing_objects['label_encoders']

                if 'source' in encoders:
                    try:
                        features['source_encoded'] = encoders['source'].transform([source])[0]
                    except ValueError:
                        features['source_encoded'] = 0

                if 'brand' in encoders:
                    try:
                        features['brand_encoded'] = encoders['brand'].transform([brand])[0]
                    except ValueError:
                        features['brand_encoded'] = 0

            # Add numerical features
            features['rating'] = rating
            features['discountoffering'] = discount

            # Add TF-IDF features
            if 'tfidf_vectorizer' in self.preprocessing_objects:
                tfidf_vec = self.preprocessing_objects['tfidf_vectorizer']
                tfidf_features = tfidf_vec.transform([mobile_name])
                for i in range(tfidf_features.shape[1]):
                    features[f'tfidf_{i}'] = tfidf_features[0, i]

            # Create feature array in correct order
            if 'feature_names' in self.preprocessing_objects:
                feature_names = self.preprocessing_objects['feature_names']
                feature_array = np.array([features.get(name, 0) for name in feature_names]).reshape(1, -1)

                # Make predictions with all available models
                predictions = {}
                for model_name, model in self.models.items():
                    pred = model.predict(feature_array)[0]
                    predictions[model_name] = max(0, pred)  # Ensure non-negative

                return predictions

        except Exception as e:
            st.error(f"Error making prediction: {e}")

        return None

# Initialize ML predictor
if SKLEARN_AVAILABLE:
    ml_predictor = MLPricePredictor()
else:
    ml_predictor = None

# ---------------- Email Notification System ----------------
def send_email(subject, body, sender, receiver, password):
    if not EMAIL_AVAILABLE:
        st.warning("Email functionality not available")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        st.success(f" Email sent: {subject}")
    except Exception as e:
        st.error(f" Email sending failed: {e}")

def notify_product_analysis(analyzer, product_name):
    """Send an email summary after analyzing a product"""
    if not EMAIL_AVAILABLE:
        return

    email_config = dashboard_config.get('email_notifications', {})
    if not email_config.get('enabled', False):
        return

    load_dotenv()
    sender = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD") 
    receiver = os.getenv("EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        st.warning("Email credentials not configured in .env")
        return

    try:
        prod = analyzer.products_df.query("`product_name`==@product_name").iloc[0]
        sdata = analyzer.get_sentiment_analysis(product_name)

        avg_sentiment = sdata["average_sentiment_score"] if sdata else 0
        sentiment_status = (
            "Negative" if avg_sentiment < 0 else
            "Neutral" if avg_sentiment < 0.2 else "Positive"
        )

        subject = f" Analysis Report: {product_name}"
        body = f"""
Product Analysis Report 

Product: {product_name}
Price: ₹{prod['price']}
Discount: {prod['discount']}%
Rating: {prod['rating']}
Sentiment Status: {sentiment_status}
Avg Sentiment Score: {avg_sentiment:.2f}

Dashboard analysis completed successfully.
        """

        send_email(subject, body, sender, receiver, password)
    except Exception as e:
        st.error(f"Error sending notification: {e}")

# ---------------- Competitor Analyzer ----------------
class CompetitorAnalyzer:
    def __init__(self):
        self.products_df = None
        self.reviews_df = None

    def load_data(self):
        """Load cleaned product & review data"""
        try:
            data_config = dashboard_config.get('data_files', {})
            products_file = data_config.get('products', 'My_docs/mobile_products.csv')
            reviews_file = data_config.get('reviews', 'My_docs/product_reviews.csv')


            if os.path.exists(products_file):
                self.products_df = pd.read_csv(products_file)
                # Rename columns to internal standard
                self.products_df.rename(columns={
                    "mobilename": "product_name",
                    "sellingprice": "price", 
                    "discountoffering": "discount",
                    "rating": "rating",
                    "productid": "product_id",
                    "source": "source"
                }, inplace=True)
            else:
                st.error(f"Missing {products_file}")
                return False

            if os.path.exists(reviews_file):
                self.reviews_df = pd.read_csv(reviews_file)
                self.reviews_df.rename(columns={
                    "mobilename": "product_name",
                    "review": "review_text",
                    "rating": "rating",
                    "reviewdate": "date",
                    "productid": "product_id", 
                    "source": "source"
                }, inplace=True)
            else:
                st.error(f"Missing {reviews_file}")
                return False

            # Clean/convert data
            self.products_df["price"] = pd.to_numeric(self.products_df["price"], errors="coerce")
            self.products_df["discount"] = pd.to_numeric(self.products_df["discount"], errors="coerce").fillna(0)
            self.products_df["rating"] = pd.to_numeric(self.products_df["rating"], errors="coerce").fillna(0)
            self.reviews_df["date"] = pd.to_datetime(self.reviews_df["date"], errors="coerce")

            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if not TEXTBLOB_AVAILABLE:
            return "neutral", 0.0

        try:
            a = TextBlob(str(text))
            p = a.sentiment.polarity
            if p > 0.1:
                return "positive", p
            elif p < -0.1:
                return "negative", p
            else:
                return "neutral", p
        except:
            return "neutral", 0.0

    def get_sentiment_analysis(self, product_name):
        """Get sentiment analysis for a product"""
        df = self.reviews_df[self.reviews_df["product_name"] == product_name].copy()
        if df.empty:
            return None

        sentiments = []
        for r in df["review_text"]:
            s, sc = self.analyze_sentiment(r)
            sentiments.append({"sentiment": s, "score": sc})

        s_df = pd.DataFrame(sentiments)
        return {
            "total_reviews": len(df),
            "sentiment_distribution": s_df["sentiment"].value_counts().to_dict(),
            "average_sentiment_score": s_df["score"].mean(),
            "reviews_data": df
        }

# ---------------- Dashboard Sections ----------------
def product_analysis(analyzer, product_name):
    st.markdown('<div class="section-header">Product Analysis</div>', unsafe_allow_html=True)

    try:
        prod = analyzer.products_df.query("`product_name` == @product_name").iloc[0]
    except IndexError:
        st.error("Product not found!")
        return

    # Basic metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"₹{int(prod['price'])}")
    c2.metric("Discount", f"{prod['discount']}%")
    c3.metric("Rating", f"{prod['rating']}/5")
    c4.metric("Source", prod["source"])

    # ML Price Predictions
    if ml_predictor and ml_predictor.models:
        st.markdown("###  ML Price Predictions")

        with st.expander("Customize prediction parameters"):
            pred_rating = st.slider("Rating", 1.0, 5.0, float(prod['rating']), 0.1)
            pred_discount = st.slider("Discount %", 0.0, 50.0, float(prod['discount']), 1.0)

        predictions = ml_predictor.predict_price(
            product_name, prod['source'], pred_rating, pred_discount
        )

        if predictions:
            pred_cols = st.columns(len(predictions))
            for i, (model_name, pred_price) in enumerate(predictions.items()):
                with pred_cols[i]:
                    accuracy_color = "green" if abs(pred_price - prod['price']) < prod['price'] * 0.1 else "orange"
                    accuracy = 100 - (abs(pred_price - prod['price']) / prod['price'] * 100)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{model_name}</h4>
                        <h3 style="color: {accuracy_color}">₹{pred_price:,.0f}</h3>
                        <p>Accuracy: {accuracy:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Sentiment Analysis
    st.subheader("Customer Sentiment")
    sdata = analyzer.get_sentiment_analysis(product_name)
    if sdata:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.pie(
                values=list(sdata["sentiment_distribution"].values()),
                names=list(sdata["sentiment_distribution"].keys()),
                title="Sentiment Distribution",
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'yellow'}
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Total Reviews", sdata["total_reviews"])
            st.metric("Avg Sentiment Score", f"{sdata['average_sentiment_score']:.2f}")

        st.markdown("### Recent Reviews")
        for _, r in sdata["reviews_data"].head(5).iterrows():
            sent, sc = analyzer.analyze_sentiment(r["review_text"])
            with st.expander(f"{r['userid']} - Rating: {r['rating']}"):
                st.write(r["review_text"])
                st.write(f"Sentiment: **{sent}** (score {sc:.2f})")
    else:
        st.info("No reviews available.")

def competitor_comparison(analyzer, product_name):
    st.markdown('<div class="section-header">Competitor Comparison</div>', unsafe_allow_html=True)

    try:
        source = analyzer.products_df.query("`product_name`==@product_name")["source"].iloc[0]
        comp = analyzer.products_df.query("source==@source and product_name!=@product_name")

        if comp.empty:
            st.info("No competitor data available.")
            return

        # Price comparison chart
        fig = px.bar(comp, x="product_name", y="price", color="price",
                     title=f"Competitor Price Comparison ({source})")
        st.plotly_chart(fig, use_container_width=True)

        # Interactive selection
        st.markdown("### Explore Competitors Near Selected Product Price")
        selected_product = st.selectbox("Select competitor product", comp["product_name"].unique())

        if selected_product:
            selected_price = comp.query("product_name==@selected_product")["price"].values[0]
            st.markdown(f"**Showing products around ₹{selected_price:,}**")

            # Price range ±20%
            lower_bound = selected_price * 0.8
            upper_bound = selected_price * 1.2

            nearby_products = analyzer.products_df.query(
                "price >= @lower_bound and price <= @upper_bound"
            ).copy()

            # Add sentiment scores
            sentiment_scores = []
            for prod in nearby_products["product_name"]:
                sentiment_info = analyzer.get_sentiment_analysis(prod)
                if sentiment_info:
                    sentiment_scores.append(sentiment_info["average_sentiment_score"])
                else:
                    sentiment_scores.append(np.nan)

            nearby_products["avg_sentiment"] = sentiment_scores
            nearby_products.sort_values(by="avg_sentiment", ascending=False, inplace=True)

            # Display table
            cols = ["product_name", "source", "price", "discount", "rating", "avg_sentiment"]
            st.dataframe(nearby_products[cols])

    except Exception as e:
        st.error(f"Error in competitor comparison: {e}")

def strategic_recommendations(analyzer, product_name):
    st.markdown('<div class="section-header">Strategic Recommendations</div>', unsafe_allow_html=True)

    try:
        prod = analyzer.products_df.query("`product_name`==@product_name").iloc[0]
        sdata = analyzer.get_sentiment_analysis(product_name)

        avg_score = sdata["average_sentiment_score"] if sdata else 0

        # Generate recommendations
        strategy_lines = []

        if prod["price"] > 50000:
            strategy_lines.append(f"- High price detected (₹{prod['price']:,}). Consider limited-time discounts or EMI options.")
        elif prod["price"] < 20000:
            strategy_lines.append(f"- Competitive price (₹{prod['price']:,}) can be leveraged with marketing campaigns.")

        if prod["discount"] < 5:
            strategy_lines.append(f"- Current discount is low ({prod['discount']}%). Increase discount to attract price-sensitive customers.")
        elif prod["discount"] > 20:
            strategy_lines.append(f"- Generous discount ({prod['discount']}%) observed. Maintain for high conversion or flash sales.")

        if avg_score < 0:
            strategy_lines.append("- Customer sentiment is negative. Investigate recurring complaints and improve product/service quality.")
        elif avg_score < 0.2:
            strategy_lines.append("- Customer sentiment is neutral. Consider improving features or adding promotional offers.")
        else:
            strategy_lines.append("- Positive sentiment! Promote product strengths in marketing campaigns.")

        # ML-based recommendations
        if ml_predictor and ml_predictor.models:
            predictions = ml_predictor.predict_price(product_name, prod['source'])
            if predictions:
                avg_prediction = np.mean(list(predictions.values()))
                if avg_prediction > prod['price'] * 1.1:
                    strategy_lines.append(f"- ML models suggest price could be increased to ₹{avg_prediction:,.0f} (10% above current).")
                elif avg_prediction < prod['price'] * 0.9:
                    strategy_lines.append(f"- ML models suggest price reduction to ₹{avg_prediction:,.0f} for better market positioning.")

        # Display recommendations
        sentiment_status = "Needs Improvement" if avg_score < 0.2 else "Good" if avg_score < 0.5 else "Excellent"
        sentiment_class = "negative-sentiment" if avg_score < 0.2 else "neutral-sentiment" if avg_score < 0.5 else "positive-sentiment"

        st.markdown(f"**Sentiment Status:** <span class='{sentiment_class}'>{sentiment_status}</span>", unsafe_allow_html=True)
        st.markdown("### Recommended Strategy")
        st.markdown("\n".join(strategy_lines))

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

def ml_model_performance():
    """Display ML model performance metrics"""
    st.markdown('<div class="section-header">ML Model Performance</div>', unsafe_allow_html=True)

    try:
        results_path = Path("model_results.csv")
        if results_path.exists():
            results_df = pd.read_csv(results_path)

            # Display metrics table
            st.subheader("Model Comparison")
            st.dataframe(results_df[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE']])

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                fig_r2 = px.bar(results_df, x='Model', y='Test_R2', 
                              title='Model R² Scores', color='Test_R2')
                st.plotly_chart(fig_r2, use_container_width=True)

            with col2:
                fig_rmse = px.bar(results_df, x='Model', y='Test_RMSE',
                                title='Model RMSE (Lower is Better)', color='Test_RMSE')
                st.plotly_chart(fig_rmse, use_container_width=True)

            # Display plots if available
            plots_dir = Path("plots")
            if plots_dir.exists():
                st.subheader("Model Visualizations")

                # Model comparison plot
                comparison_plot = plots_dir / "model_comparison.png"
                if comparison_plot.exists():
                    st.image(str(comparison_plot), caption="Model Comparison")

                # Actual vs Predicted plot
                pred_plot = plots_dir / "actual_vs_predicted.png"
                if pred_plot.exists():
                    st.image(str(pred_plot), caption="Actual vs Predicted Prices")

        else:
            st.info("No ML model results found. Run ML training first.")

    except Exception as e:
        st.error(f"Error displaying ML performance: {e}")

# ---------------- Main App ----------------
def main():
    st.markdown('<div class="main-header">E-Commerce Competitor Strategy Dashboard</div>', unsafe_allow_html=True)

    analyzer = CompetitorAnalyzer()
    if not analyzer.load_data():
        st.stop()

    # Sidebar navigation
    features = dashboard_config.get('features', {})
    nav_options = []

    if features.get('product_analysis', True):
        nav_options.append("Product Analysis")
    if features.get('competitor_comparison', True):
        nav_options.append("Competitor Comparison")
    if features.get('strategic_recommendations', True):
        nav_options.append("Strategic Recommendations")
    if features.get('ml_predictions', True):
        nav_options.append("ML Model Performance")

    section = st.sidebar.radio("Navigate", nav_options)

    if section != "ML Model Performance":
        product = st.sidebar.selectbox("Select Product", analyzer.products_df["product_name"].unique())

    # Execute selected section
    if section == "Product Analysis":
        product_analysis(analyzer, product)
    elif section == "Competitor Comparison":
        competitor_comparison(analyzer, product)
    elif section == "Strategic Recommendations":
        strategic_recommendations(analyzer, product)
    elif section == "ML Model Performance":
        ml_model_performance()

    # Email notification button
    if (dashboard_config.get('email_notifications', {}).get('enabled', False) and 
        section != "ML Model Performance" and st.sidebar.button(" Send Report to Email")):
        notify_product_analysis(analyzer, product)

if __name__ == "__main__":
    main()
