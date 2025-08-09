from app import app, db

# Importe seus modelos aqui para que o SQLAlchemy saiba quais tabelas criar
from models import Customer, Conversation, Sale, Product, AILearningData, WhatsAppWebhook

with app.app_context():
    # Cria todas as tabelas que estão definidas nos seus modelos
    db.create_all()
    print("Banco de dados inicializado e tabelas criadas com sucesso!")
