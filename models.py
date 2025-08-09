from app import db
from datetime import datetime
from sqlalchemy import Text, DateTime, Boolean, Float, Integer, String

class Product(db.Model):
    """Product information and niche details"""
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    niche = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=False)
    target_audience = db.Column(db.Text, nullable=False)
    key_benefits = db.Column(db.Text, nullable=False)  # JSON string of benefits list
    sales_approach = db.Column(db.String(50), nullable=False, default='consultivo')  # consultivo, escassez, emocional, racional
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    whatsapp_number = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100))
    first_contact = db.Column(DateTime, default=datetime.utcnow)
    last_interaction = db.Column(DateTime, default=datetime.utcnow)
    total_interactions = db.Column(Integer, default=0)
    purchased = db.Column(Boolean, default=False)
    purchase_date = db.Column(DateTime)
    
    # Relationships
    conversations = db.relationship('Conversation', backref='customer', lazy=True, cascade='all, delete-orphan')
    sales = db.relationship('Sale', backref='customer', lazy=True, cascade='all, delete-orphan')

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    message_type = db.Column(db.String(20), nullable=False)  # 'incoming' or 'outgoing'
    message_content = db.Column(Text, nullable=False)
    timestamp = db.Column(DateTime, default=datetime.utcnow)
    ai_strategy = db.Column(db.String(100))  # Strategy used by AI for this message
    sentiment_score = db.Column(Float)  # Customer sentiment analysis
    
class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=True)  # Reference to Product
    product_name = db.Column(db.String(200), nullable=False)  # Keep for backward compatibility
    sale_amount = db.Column(Float, nullable=False)
    sale_date = db.Column(DateTime, default=datetime.utcnow)
    conversation_messages = db.Column(Integer, default=0)  # Number of messages in the conversation that led to sale
    
class AILearningData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    strategy_name = db.Column(db.String(100), nullable=False)
    success_count = db.Column(Integer, default=0)
    total_attempts = db.Column(Integer, default=0)
    success_rate = db.Column(Float, default=0.0)
    last_updated = db.Column(DateTime, default=datetime.utcnow)
    
    # Conversation context that led to success/failure
    context_keywords = db.Column(Text)  # JSON string of keywords that were present
    customer_sentiment = db.Column(Float)
    message_sequence = db.Column(Text)  # JSON string of message types in sequence

class WhatsAppWebhook(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    webhook_data = db.Column(Text, nullable=False)  # Raw webhook JSON
    processed = db.Column(Boolean, default=False)
    timestamp = db.Column(DateTime, default=datetime.utcnow)
    error_message = db.Column(Text)
