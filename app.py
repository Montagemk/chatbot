import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "whatsapp-ai-sales-agent-2024")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "postgresql://admin:iHFXySgGm3lxewDc10FwIiBusuDGHv50@dpg-d2b69gidbo4c73ahs1qg-a/banco_de_dados_vendas")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models to ensure tables are created
    import models
    db.create_all()

# Add custom Jinja2 filter for JSON parsing
import json

@app.template_filter('from_json')
def from_json_filter(value):
    """Parse JSON string to Python object"""
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return []

# Import and register routes
from routes import *

# Initialize handlers after app context is ready
with app.app_context():
    from routes import init_handlers
    init_handlers()
