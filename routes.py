from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app, db
from models import Customer, Conversation, Sale, AILearningData, Product
from ai_agent import AIAgent
from reinforcement_learning import ReinforcementLearner
from datetime import datetime, timedelta
import json
import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)

ai_agent = None
learner = None

SECRET_API_KEY = os.environ.get("WEBCHAT_API_KEY", "sua-chave-secreta-deve-ser-trocada")

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-Key') != SECRET_API_KEY:
            logger.warning("Tentativa de acesso ao webhook com API Key inválida.")
            return jsonify({"error": "Chave de API inválida ou ausente"}), 403
        return f(*args, **kwargs)
    return decorated_function

def init_handlers():
    global ai_agent, learner
    if ai_agent is None:
        ai_agent = AIAgent()
    if learner is None:
        learner = ReinforcementLearner()

@app.route('/webhook', methods=['POST'])
@require_api_key
def web_chat_webhook():
    try:
        if ai_agent is None: init_handlers()
        webhook_data = request.get_json()
        message_content = webhook_data.get('message', '')
        sender_id = webhook_data.get('sender')
        
        if not sender_id or not message_content:
            return jsonify([{"text": "Dados da requisição incompletos."}]), 400

        customer = Customer.query.filter_by(whatsapp_number=sender_id).first()
        if not customer:
            customer = Customer(whatsapp_number=sender_id, name=f"Visitante_{sender_id[:6]}")
            db.session.add(customer)
            db.session.flush()
        
        customer.last_interaction = datetime.utcnow()
        customer.total_interactions = (customer.total_interactions or 0) + 1
        
        conversation_history = Conversation.query.filter_by(customer_id=customer.id).order_by(Conversation.timestamp.desc()).limit(10).all()
        conversation_history.reverse() # Ordena da mais antiga para a mais recente
        conversation_dict = [{'message_type': conv.message_type, 'message_content': conv.message_content} for conv in conversation_history]
        
        # A MÁGICA DO NOVO FUNIL ACONTECE AQUI
        # Passamos o cliente inteiro para a IA, para que ela saiba o estado do funil
        structured_response = ai_agent.generate_response(customer, message_content, conversation_dict)
        
        customer_analysis = structured_response.get("analysis", {})
        ai_response_text = structured_response.get("response", "Desculpe, não consegui processar sua mensagem.")
        
        # Atualiza o estado do funil e o produto selecionado, se a IA os retornou
        new_funnel_state = structured_response.get("funnel_state_update")
        if new_funnel_state:
            customer.funnel_state = new_funnel_state

        product_id_to_select = structured_response.get("product_id_to_select")
        if product_id_to_select:
            customer.selected_product_id = product_id_to_select

        incoming_conversation = Conversation(
            customer_id=customer.id, message_type='incoming',
            message_content=message_content, sentiment_score=customer_analysis.get('sentiment', 0.0)
        )
        outgoing_conversation = Conversation(
            customer_id=customer.id, message_type='outgoing', message_content=ai_response_text
        )
        db.session.add_all([incoming_conversation, outgoing_conversation])
        db.session.commit()
        
        logger.info(f"Mensagem de {sender_id} (estado: {customer.funnel_state}) processada.")
        return jsonify([{"recipient_id": sender_id, "text": ai_response_text}]), 200

    except Exception as e:
        logger.error(f"Erro ao processar mensagem do chat web: {e}", exc_info=True)
        db.session.rollback()
        return jsonify([{"text": "Desculpe, ocorreu um erro no servidor. Tente novamente."}]), 500

@app.route('/')
def dashboard():
    # ... (código inalterado)
    return render_template('dashboard.html', ...)

@app.route('/conversations')
def conversations():
    # ... (código inalterado)
    return render_template('conversations.html', ...)

@app.route('/products')
def products():
    all_products = Product.query.order_by(Product.created_at.desc()).all()
    return render_template('products.html', products=all_products)

# --- FUNÇÃO ATUALIZADA ---
@app.route('/products/new', methods=['GET', 'POST'])
def new_product():
    if request.method == 'POST':
        try:
            benefits_text = request.form.get('key_benefits', '')
            benefits_list = [benefit.strip() for benefit in benefits_text.split('\n') if benefit.strip()]
            
            new_prod = Product(
                name=request.form.get('name'),
                niche=request.form.get('niche'),
                original_price=float(request.form.get('original_price')) if request.form.get('original_price') else None,
                price=float(request.form.get('price', 0)),
                description=request.form.get('description'),
                target_audience=request.form.get('target_audience'),
                sales_approach=request.form.get('sales_approach', 'consultivo'),
                key_benefits=json.dumps(benefits_list),
                payment_link=request.form.get('payment_link'),
                product_image_url=request.form.get('product_image_url'),
                free_group_link=request.form.get('free_group_link'),
                specialist_name=request.form.get('specialist_name'),
                specialist_social_proof=request.form.get('specialist_social_proof'),
                testimonials_link=request.form.get('testimonials_link')
            )
            db.session.add(new_prod)
            db.session.commit()
            flash(f'Produto "{new_prod.name}" criado com sucesso!', 'success')
            return redirect(url_for('products'))
        except Exception as e:
            flash(f'Erro ao criar produto: {e}', 'error')
            db.session.rollback()
    return render_template('new_product.html')

# --- FUNÇÃO ATUALIZADA ---
@app.route('/products/<int:product_id>/edit', methods=['GET', 'POST'])
def edit_product(product_id):
    product = Product.query.get_or_404(product_id)
    if request.method == 'POST':
        try:
            benefits_text = request.form.get('key_benefits', '')
            benefits_list = [benefit.strip() for benefit in benefits_text.split('\n') if benefit.strip()]

            product.name = request.form.get('name')
            product.niche = request.form.get('niche')
            product.original_price = float(request.form.get('original_price')) if request.form.get('original_price') else None
            product.price = float(request.form.get('price', 0))
            product.description = request.form.get('description')
            product.target_audience = request.form.get('target_audience')
            product.sales_approach = request.form.get('sales_approach', 'consultivo')
            product.key_benefits = json.dumps(benefits_list)
            product.payment_link = request.form.get('payment_link')
            product.product_image_url = request.form.get('product_image_url')
            product.free_group_link = request.form.get('free_group_link')
            product.is_active = request.form.get('is_active') == 'on'
            
            product.specialist_name = request.form.get('specialist_name')
            product.specialist_social_proof = request.form.get('specialist_social_proof')
            product.testimonials_link = request.form.get('testimonials_link')
            
            product.updated_at = datetime.utcnow()
            db.session.commit()
            flash(f'Produto "{product.name}" atualizado com sucesso!', 'success')
            return redirect(url_for('products'))
        except Exception as e:
            flash(f'Erro ao atualizar produto: {e}', 'error')
            db.session.rollback()
            
    benefits_text = '\n'.join(json.loads(product.key_benefits)) if product.key_benefits else ""
    return render_template('edit_product.html', product=product, benefits_text=benefits_text)

# --- NOVA FUNÇÃO ADICIONADA ---
@app.route('/products/<int:product_id>/delete', methods=['POST'])
def delete_product(product_id):
    try:
        product = Product.query.get_or_404(product_id)
        product.is_active = False
        product.updated_at = datetime.utcnow()
        db.session.commit()
        flash(f'Produto "{product.name}" foi desativado com sucesso!', 'success')
    except Exception as e:
        flash(f'Erro ao desativar produto: {str(e)}', 'error')
        db.session.rollback()
    return redirect(url_for('products'))

# ... (resto das rotas como /niches, /customer, etc. permanecem inalteradas)
