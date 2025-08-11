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
        
        incoming_conversation = Conversation(
            customer_id=customer.id, message_type='incoming',
            message_content=message_content
        )
        db.session.add(incoming_conversation)
        db.session.commit()

        conversation_history = Conversation.query.filter_by(customer_id=customer.id).order_by(Conversation.timestamp.desc()).limit(10).all()
        conversation_history.reverse()
        conversation_dict = [{'message_type': conv.message_type, 'message_content': conv.message_content} for conv in conversation_history]
        
        structured_response = ai_agent.generate_response(customer, conversation_dict)
        
        customer_analysis = structured_response.get("analysis", {})
        ai_response_text = structured_response.get("response", "Desculpe, não consegui processar sua mensagem.")
        
        outgoing_conversation = Conversation(
            customer_id=customer.id, message_type='outgoing', message_content=ai_response_text,
            sentiment_score=customer_analysis.get('sentiment', 0.0)
        )
        db.session.add(outgoing_conversation)

        new_funnel_state = structured_response.get("funnel_state_update")
        if new_funnel_state:
            customer.funnel_state = new_funnel_state

        product_id_to_select = structured_response.get("product_id_to_select")
        if product_id_to_select:
            customer.selected_product_id = int(product_id_to_select)
        
        db.session.commit()

        final_response_text = ai_response_text

        if new_funnel_state == 'Specialist_Intro':
            logger.info(f"Handoff detectado. Gerando primeira mensagem do especialista para o cliente {customer.id}.")
            specialist_response_dict = ai_agent.generate_response(customer, []) 
            
            specialist_text = specialist_response_dict.get("response", "Olá! Sou o especialista e estou aqui para ajudar.")
            final_response_text += f"\n\n{specialist_text}"

            specialist_outgoing = Conversation(
                customer_id=customer.id, message_type='outgoing', message_content=specialist_text
            )
            db.session.add(specialist_outgoing)

            final_state = specialist_response_dict.get("funnel_state_update")
            if final_state:
                customer.funnel_state = final_state
            
            db.session.commit()

        logger.info(f"Mensagem de {sender_id} (estado: {customer.funnel_state}) processada.")
        return jsonify([{"recipient_id": sender_id, "text": final_response_text}]), 200

    except Exception as e:
        logger.error(f"Erro ao processar mensagem do chat web: {e}", exc_info=True)
        db.session.rollback()
        return jsonify([{"text": "Desculpe, ocorreu um erro no servidor. Tente novamente."}]), 500

@app.route('/')
def dashboard():
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
        return render_template('dashboard.html', total_customers=total_customers, total_conversations=total_conversations, total_sales=total_sales, total_revenue=total_revenue, recent_customers=recent_customers, recent_sales=recent_sales, conversion_rate=conversion_rate, learning_stats=learning_stats)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return render_template('dashboard.html', error=str(e))

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
    try:
        page = request.args.get('page', 1, type=int)
        customers = Customer.query.order_by(Customer.last_interaction.desc()).paginate(
            page=page, per_page=20, error_out=False
        )
        return render_template('conversations.html', customers=customers, now=datetime.utcnow())
    except Exception as e:
        logger.error(f"Error loading conversations page: {e}")
        return render_template('conversations.html', error=str(e))

@app.route('/customer/<int:customer_id>')
def customer_detail(customer_id):
    customer = Customer.query.get_or_404(customer_id)
    conversations = Conversation.query.filter_by(customer_id=customer_id).order_by(Conversation.timestamp.asc()).all()
    return render_template('customer_detail.html', customer=customer, conversations=conversations)

@app.route('/analytics')
def analytics():
    if learner is None: init_handlers()
    learning_stats = learner.get_learning_statistics()
    # Adicione a lógica para buscar daily_sales e avg_sentiment se necessário para o template
    return render_template('analytics.html', learning_stats=learning_stats)

@app.route('/api/learning_stats')
def api_learning_stats():
    if learner is None: init_handlers()
    stats = learner.get_learning_statistics()
    return jsonify(stats)

@app.route('/products')
def products():
    all_products = Product.query.order_by(Product.created_at.desc()).all()
    return render_template('products.html', products=all_products)

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
            product.sales_approach = request.form.get('sales_approach')
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
            
    benefits_text = '\n'.join(json.loads(product.key_benefits)) if product.key_benefits and product.key_benefits.strip() else ""
    return render_template('edit_product.html', product=product, benefits_text=benefits_text)

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

@app.route('/niches')
def niches():
    try:
        from sqlalchemy import func
        
        niche_query_result = db.session.query(
            Product.niche,
            func.count(Product.id).label('total_products'),
            func.count(Sale.id).label('total_sales'),
            func.sum(Sale.sale_amount).label('total_revenue'),
            func.avg(Sale.sale_amount).label('avg_sale_value')
        ).outerjoin(Sale, Product.id == Sale.product_id
        ).filter(Product.is_active == True
        ).group_by(Product.niche).all()
        
        niche_stats = [
            {
                "niche": row.niche,
                "total_products": row.total_products,
                "total_sales": row.total_sales,
                "total_revenue": row.total_revenue,
                "avg_sale_value": row.avg_sale_value,
            }
            for row in niche_query_result
        ]
        
        return render_template('niches.html', niche_stats=niche_stats)
    except Exception as e:
        logger.error(f"Error loading niches page: {e}")
        return render_template('niches.html', error=str(e))
