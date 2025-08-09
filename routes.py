from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app, db
from models import Customer, Conversation, Sale, AILearningData, Product
from ai_agent import AIAgent
from reinforcement_learning import ReinforcementLearner
from datetime import datetime, timedelta
import json
import logging
import os

logger = logging.getLogger(__name__)

ai_agent = None
learner = None

def init_handlers():
    """Initialize handlers within application context"""
    global ai_agent, learner
    if ai_agent is None:
        ai_agent = AIAgent()
    if learner is None:
        learner = ReinforcementLearner()

@app.route('/')
def dashboard():
    """Main dashboard with overview statistics"""
    try:
        if learner is None:
            init_handlers()
        total_customers = Customer.query.count()
        total_conversations = Conversation.query.count()
        total_sales = Sale.query.count()
        total_revenue = db.session.query(db.func.sum(Sale.sale_amount)).scalar() or 0
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_customers = Customer.query.filter(Customer.first_contact >= week_ago).count()
        recent_sales = Sale.query.filter(Sale.sale_date >= week_ago).count()
        conversion_rate = (total_sales / total_customers * 100) if total_customers > 0 else 0
        learning_stats = learner.get_learning_statistics()
        
        return render_template('dashboard.html', 
                             total_customers=total_customers,
                             total_conversations=total_conversations,
                             total_sales=total_sales,
                             total_revenue=total_revenue,
                             recent_customers=recent_customers,
                             recent_sales=recent_sales,
                             conversion_rate=conversion_rate,
                             learning_stats=learning_stats)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return render_template('dashboard.html', error=str(e))

@app.route('/webhook', methods=['POST'])
def web_chat_webhook():
    """
    Handle incoming messages from the web chat interface.
    This is the main entry point for customer messages.
    """
    try:
        # Garante que os handlers foram inicializados
        if ai_agent is None:
            init_handlers()
            
        webhook_data = request.get_json()
        message_content = webhook_data.get('message', '')
        sender_id = webhook_data.get('sender', 'web_user')
        
        if not message_content:
            return jsonify([{"text": "Mensagem vazia."}]), 400

        # --- LÓGICA ATUALIZADA: BUSCA TODOS OS PRODUTOS ATIVOS ---
        # Em vez de pegar só o primeiro, pega a lista completa para a IA decidir o que fazer.
        available_products = Product.query.filter_by(is_active=True).all()
        if not available_products:
            logger.warning("Nenhum produto ativo encontrado no banco de dados.")
            return jsonify([{"text": "Olá! No momento, estamos atualizando nosso catálogo. Volte em breve!"}]), 200

        # Encontra ou cria o cliente
        customer = Customer.query.filter_by(whatsapp_number=sender_id).first()
        if not customer:
            customer = Customer(whatsapp_number=sender_id, name="Web Chat User")
            db.session.add(customer)
            db.session.flush() # Para obter o ID do cliente antes do commit
        
        # Atualiza os dados do cliente
        customer.last_interaction = datetime.utcnow()
        customer.total_interactions = (customer.total_interactions or 0) + 1
        
        # Obtém o histórico da conversa
        conversation_history = Conversation.query.filter_by(
            customer_id=customer.id
        ).order_by(Conversation.timestamp.asc()).limit(20).all()
        conversation_dict = [{'message_type': conv.message_type, 'message_content': conv.message_content} for conv in conversation_history]
        
        # Analisa a intenção do cliente (pode ser aprimorado no futuro)
        customer_analysis = ai_agent.analyze_customer_intent(message_content, conversation_dict)
        
        # --- CHAMADA ATUALIZADA: PASSA A LISTA DE PRODUTOS PARA A IA ---
        ai_response = ai_agent.generate_response(
            message_content, customer_analysis, conversation_dict, available_products=available_products
        )
        
        # Salva a conversa no banco de dados
        incoming_conversation = Conversation(customer_id=customer.id, message_type='incoming', message_content=message_content)
        outgoing_conversation = Conversation(customer_id=customer.id, message_type='outgoing', message_content=ai_response) # Opcional: salvar a estratégia usada
        db.session.add_all([incoming_conversation, outgoing_conversation])
        db.session.commit()
        
        logger.info(f"Mensagem de {sender_id} processada com sucesso.")
        return jsonify([{"recipient_id": sender_id, "text": ai_response}]), 200
            
    except Exception as e:
        logger.error(f"Erro ao processar mensagem do chat web para {sender_id}: {e}")
        db.session.rollback()
        return jsonify([{"text": "Desculpe, ocorreu um erro no servidor. Tente novamente."}]), 500

#
# O restante do arquivo (rotas do painel de admin) não precisa de alterações.
#
@app.route('/simulate_sale', methods=['POST'])
def simulate_sale():
    try:
        if learner is None: init_handlers()
        data = request.get_json()
        customer_id = data.get('customer_id')
        if not customer_id: return jsonify({"error": "customer_id é obrigatório"}), 400
        customer = Customer.query.get(customer_id)
        if not customer: return jsonify({"error": "Cliente não encontrado"}), 404
        product = Product.query.filter_by(is_active=True).first()
        if not product: return jsonify({"error": "Nenhum produto ativo encontrado para simular a venda"}), 404
        latest_conv = Conversation.query.filter_by(customer_id=customer_id, message_type='outgoing').order_by(Conversation.timestamp.desc()).first()
        strategy_used = latest_conv.ai_strategy if latest_conv and latest_conv.ai_strategy else "consultivo"
        sale = Sale(customer_id=customer_id, product_id=product.id, product_name=product.name, sale_amount=product.price, conversation_messages=customer.total_interactions)
        db.session.add(sale)
        customer.purchased = True
        customer.purchase_date = datetime.utcnow()
        learner.record_success(customer_id, strategy_used, customer.total_interactions)
        db.session.commit()
        return jsonify({"message": f"Venda do produto '{product.name}' registrada com sucesso", "strategy": strategy_used})
    except Exception as e:
        logger.error(f"Erro ao simular venda: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/conversations')
def conversations():
    page = request.args.get('page', 1, type=int)
    customers = Customer.query.order_by(Customer.last_interaction.desc()).paginate(page=page, per_page=20, error_out=False)
    return render_template('conversations.html', customers=customers)

@app.route('/customer/<int:customer_id>')
def customer_detail(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    conversations = Conversation.query.filter_by(customer_id=customer_id).order_by(Conversation.timestamp.asc()).all()
    return render_template('customer_detail.html', customer=customer, conversations=conversations)

@app.route('/analytics')
def analytics():
    if learner is None: init_handlers()
    learning_stats = learner.get_learning_statistics()
    return render_template('analytics.html', learning_stats=learning_stats)

@app.route('/api/learning_stats')
def api_learning_stats():
    if learner is None: init_handlers()
    stats = learner.get_learning_statistics()
    return jsonify(stats)

@app.route('/products')
def products():
    products = Product.query.filter_by(is_active=True).order_by(Product.created_at.desc()).all()
    return render_template('products.html', products=products)

@app.route('/products/new', methods=['GET', 'POST'])
def new_product():
    if request.method == 'POST':
        try:
            # ... (código de criação do produto)
            flash('Produto criado com sucesso!', 'success')
            return redirect(url_for('products'))
        except Exception as e:
            flash(f'Erro ao criar produto: {e}', 'error')
            db.session.rollback()
    return render_template('new_product.html')

@app.route('/products/<int:product_id>/edit', methods=['GET', 'POST'])
def edit_product(product_id):
    product = Product.query.get_or_404(product_id)
    if request.method == 'POST':
        try:
            # ... (código de edição do produto)
            flash('Produto atualizado com sucesso!', 'success')
            return redirect(url_for('products'))
        except Exception as e:
            flash(f'Erro ao atualizar produto: {e}', 'error')
            db.session.rollback()
    # ... (código para exibir o formulário de edição)
    return render_template('edit_product.html', product=product)

@app.route('/products/<int:product_id>/delete', methods=['POST'])
def delete_product(product_id):
    # ... (código de deleção do produto)
    return redirect(url_for('products'))

@app.route('/niches')
def niches():
    # ... (código da página de nichos)
    return render_template('niches.html')
