from app import app, db

# Importa apenas os modelos necessários para a aplicação
from models import Customer, Conversation, Sale, Product, AILearningData

with app.app_context():
    # Cria todas as tabelas que estão definidas nos seus modelos
    db.create_all()
    print("Banco de dados inicializado e tabelas criadas com sucesso!")
