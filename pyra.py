from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from datetime import datetime, timedelta
import uuid
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pyra AI Customer Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    customer_id: str
    engagement_score: float
    support_tickets: int
    purchase_history: float
    feedback_sentiment: float
    days_since_last_interaction: int


class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    customer_context: dict = None


class RecommendationRequest(BaseModel):
    customer_id: str
    interaction_history: list = []


class CustomerDataGenerator:
    def __init__(self):
        self.customers = {}
        self.products = [
            "Premium Subscription", "Basic Plan", "Enterprise Suite",
            "Mobile App", "Web Dashboard", "API Access", "Consulting Hours",
            "Training Program", "Support Package", "Custom Integration"
        ]
        self.issues = [
            "billing payment invoice refund charge",
            "technical bug error not working issue problem",
            "account login password signup verification",
            "feature request how-to tutorial guide",
            "general help support contact question"
        ]
        self.generate_customer_base()
        self.train_models()

    def generate_customer_base(self):
        np.random.seed(42)
        for i in range(1000):
            customer_id = f"CUST_{i:05d}"
            self.customers[customer_id] = {
                'engagement_score': np.random.beta(2, 2),
                'support_tickets': np.random.poisson(3),
                'purchase_history': np.random.exponential(1000),
                'feedback_sentiment': np.random.normal(0.7, 0.2),
                'days_since_last_interaction': np.random.exponential(30),
                'preferred_products': np.random.choice(self.products, 3, replace=False).tolist(),
                'churn_risk': np.random.beta(2, 5)
            }

    def train_models(self):
        X = np.array([[c['engagement_score'], c['support_tickets'], c['purchase_history'],
                      c['feedback_sentiment'], c['days_since_last_interaction']]
                     for c in self.customers.values()])
        y_churn = np.array(
            [1 if c['churn_risk'] > 0.7 else 0 for c in self.customers.values()])
        y_satisfaction = np.array([c['feedback_sentiment']
                                  for c in self.customers.values()])

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.churn_model = GradientBoostingClassifier(
            n_estimators=100, random_state=42)
        self.churn_model.fit(X_scaled, y_churn)

        self.satisfaction_model = GradientBoostingRegressor(
            n_estimators=100, random_state=42)
        self.satisfaction_model.fit(X_scaled, y_satisfaction)

        self.tfidf = TfidfVectorizer()
        issue_vectors = self.tfidf.fit_transform(self.issues)

    def predict_churn(self, customer_data):
        features = np.array([[customer_data['engagement_score'], customer_data['support_tickets'],
                            customer_data['purchase_history'], customer_data['feedback_sentiment'],
                            customer_data['days_since_last_interaction']]])
        features_scaled = self.scaler.transform(features)
        probability = self.churn_model.predict_proba(features_scaled)[0][1]
        return float(probability)

    def predict_satisfaction(self, customer_data):
        features = np.array([[customer_data['engagement_score'], customer_data['support_tickets'],
                            customer_data['purchase_history'], customer_data['feedback_sentiment'],
                            customer_data['days_since_last_interaction']]])
        features_scaled = self.scaler.transform(features)
        prediction = self.satisfaction_model.predict(features_scaled)[0]
        return max(0.0, min(1.0, float(prediction)))

    def get_recommendations(self, customer_id, message=""):
        customer = self.customers.get(
            customer_id, self.customers["CUST_00000"])
        preferred = customer['preferred_products']

        if message:
            message_vec = self.tfidf.transform([message])
            issue_vecs = self.tfidf.transform(self.issues)
            similarities = cosine_similarity(message_vec, issue_vecs)[0]
            best_match_idx = np.argmax(similarities)

            if similarities[best_match_idx] > 0.3:
                if best_match_idx == 0:
                    return ["Billing Specialist Contact", "Payment Plan Options", "Invoice Review"]
                elif best_match_idx == 1:
                    return ["Technical Support Escalation", "Bug Fix Timeline", "Workaround Solution"]
                elif best_match_idx == 2:
                    return ["Account Recovery Process", "Password Reset", "Security Verification"]
                elif best_match_idx == 3:
                    return ["Feature Documentation", "Video Tutorials", "Product Training"]

        return preferred + ["Personalized Support", "Proactive Monitoring"]


data_generator = CustomerDataGenerator()
chat_sessions = {}


class ChatBot:
    def __init__(self):
        self.responses = {
            'billing': "I understand you have billing questions. Our billing team can help with invoices, payments, and refunds. Would you like me to connect you?",
            'technical': "I see you're experiencing technical issues. Let me gather some details to help our technical team resolve this quickly.",
            'account': "For account-related issues, I can help with login problems, password resets, or account settings. What specifically do you need help with?",
            'product': "I'd be happy to help you learn more about our products. We have tutorials, documentation, and training available.",
            'general': "Thank you for reaching out! I'm here to help with any questions about our services and support options."
        }

    def generate_response(self, message, customer_context=None):
        message_lower = message.lower()

        if any(word in message_lower for word in ['bill', 'payment', 'invoice', 'refund', 'charge']):
            intent = 'billing'
        elif any(word in message_lower for word in ['bug', 'error', 'not working', 'issue', 'problem', 'crash']):
            intent = 'technical'
        elif any(word in message_lower for word in ['login', 'password', 'account', 'sign up', 'verification']):
            intent = 'account'
        elif any(word in message_lower for word in ['how to', 'tutorial', 'guide', 'feature', 'learn']):
            intent = 'product'
        else:
            intent = 'general'

        base_response = self.responses[intent]

        if customer_context and customer_context.get('satisfaction', 0) < 0.5:
            base_response += " I see you've had some challenges recently - I'll make sure you get priority support."

        if customer_context and customer_context.get('churn_risk', 0) > 0.7:
            base_response += " As a valued customer, I want to ensure you're completely satisfied with our service."

        return {
            "response": base_response,
            "intent": intent,
            "recommendations": data_generator.get_recommendations("CUST_00000", message),
            "timestamp": datetime.utcnow().isoformat()
        }


chatbot = ChatBot()


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict")
async def predict_customer_metrics(request: PredictionRequest):
    try:
        customer_data = {
            'engagement_score': request.engagement_score,
            'support_tickets': request.support_tickets,
            'purchase_history': request.purchase_history,
            'feedback_sentiment': request.feedback_sentiment,
            'days_since_last_interaction': request.days_since_last_interaction
        }

        churn_risk = data_generator.predict_churn(customer_data)
        satisfaction = data_generator.predict_satisfaction(customer_data)
        recommendations = data_generator.get_recommendations(
            request.customer_id)

        risk_level = "high" if churn_risk > 0.7 else "medium" if churn_risk > 0.4 else "low"

        next_best_actions = []
        if churn_risk > 0.7:
            next_best_actions = ["Proactive outreach",
                                 "Personalized discount", "Account review"]
        elif satisfaction < 0.6:
            next_best_actions = ["Follow-up survey",
                                 "Support check-in", "Feature guidance"]
        else:
            next_best_actions = ["Upsell opportunity",
                                 "Feedback request", "Referral program"]

        return {
            "customer_id": request.customer_id,
            "churn_risk": churn_risk,
            "satisfaction_score": satisfaction,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "next_best_actions": next_best_actions,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        if not request.session_id:
            request.session_id = str(uuid.uuid4())

        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = {
                'start_time': datetime.utcnow(),
                'message_count': 0,
                'context': request.customer_context or {}
            }

        session = chat_sessions[request.session_id]
        session['message_count'] += 1

        response = chatbot.generate_response(
            request.message, session['context'])

        session['last_interaction'] = datetime.utcnow()
        session['context']['last_intent'] = response['intent']

        return {
            "session_id": request.session_id,
            "response": response['response'],
            "intent": response['intent'],
            "recommendations": response['recommendations'],
            "message_count": session['message_count'],
            "timestamp": response['timestamp']
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")


@app.post("/recommend")
async def get_personalized_recommendations(request: RecommendationRequest):
    try:
        recommendations = data_generator.get_recommendations(
            request.customer_id)

        interaction_analysis = "standard"
        if request.interaction_history:
            recent_sentiment = np.mean(
                [i.get('sentiment', 0.5) for i in request.interaction_history[-5:]])
            if recent_sentiment < 0.3:
                interaction_analysis = "needs_attention"
            elif recent_sentiment > 0.8:
                interaction_analysis = "high_value"

        return {
            "customer_id": request.customer_id,
            "personalized_recommendations": recommendations,
            "interaction_analysis": interaction_analysis,
            "recommendation_confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(
            status_code=500, detail="Recommendation generation failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
