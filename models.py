from app import db
from datetime import datetime
from sqlalchemy import Text, DateTime, Boolean, Float, Integer, String

class Product(db.Model):
    """Product information and niche details"""
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    niche = db.Column(db.String(100), nullable=False)
    
    original_price = db.Column(db.Float, nullable=True)
    price = db.Column(db.Float, nullable=False)

    description = db.Column(db.Text, nullable=False)
    target_audience = db.Column(db.Text, nullable=False)
    key_benefits = db.Column(db.Text, nullable=False)
    sales_approach = db.Column(db.String(50), nullable=False, default='consultivo')
    
    payment_link = db.Column(db.String(500), nullable=True)
    product_image_url = db.Column(db.String(500), nullable=True)
    free_group_link = db.Column(db.String(500), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    # --- ALTERAÇÃO APLICADA AQUI ---
    # Aumentamos o limite de 20 para 100 para acomodar os IDs únicos do site.
    whatsapp_number = db.Column(db.String(100), unique=True, nullable=False)
    # --- FIM DA ALTERAÇÃO ---
    
    name = db.Column(db.String(100))
    first_contact = db.Column(db.DateTime, default=datetime.utcnow)
    last_interaction = db.Column(db.DateTime, default=datetime.utcnow)
    total_interactions = db.Column(db.Integer, default=0)
    purchased = db.Column(db.Boolean, default=False)
    purchase_date = db.Column(db.DateTime)
    
    # Relationships
    conversations = db.relationship('Conversation', backref='customer', lazy=True, cascade='all, delete-orphan')
    sales = db.relationship('Sale', backref='customer', lazy=True, cascade='all, delete-orphan')

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    message_type = db.Column(db.String(20), nullable=False)
    message_content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ai_strategy = db.Column(db.String(100))
    sentiment_score = db.Column(db.Float)
    
class Sale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=True)
    product_name = db.Column(db.String(200), nullable=
