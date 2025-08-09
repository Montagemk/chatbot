from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app, db
from models import Customer, Conversation, Sale, WhatsAppWebhook, AILearningData, Product
from whatsapp_handler import WhatsAppHandler
from ai_agent import AIAgent
from reinforcement_learning import ReinforcementLearner
from datetime import datetime, timedelta
import json
import logging
import os

logger = logging.getLogger(__name__)

# Initialize handlers - these will be initialized when app starts
whatsapp = None
ai_agent = None
learner = None

def init_handlers():
    """Initialize handlers within application context"""
    global whatsapp, ai_agent, learner
    # O WhatsAppHandler não é mais necessário para a integração no site
    # whatsapp = WhatsAppHandler()
    ai_agent = AIAgent()
    learner = ReinforcementLearner()

@app.route('/')
def dashboard():
    """Main dashboard with overview statistics"""
    try:
        # Ensure handlers are initialized
        if learner is None:
            init_handlers()
            
        # Get basic statistics
        total_customers = Customer.query.count()
        total_conversations = Conversation.query.count()
        total_sales = Sale.query.count()
        
        # Sales metrics
        total_revenue = db.session.query(db.func.sum(Sale.sale_amount)).scalar() or 0
        
        # Recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_customers = Customer.query.filter(Customer.first_contact >= week_ago).count()
        recent_sales = Sale.query.filter(Sale.sale_date >= week_ago).count()
        
        # Conversion rate
        conversion_rate = (total_sales / total_customers * 100) if total_customers > 0 else 0
        
        # AI Learning statistics
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

@app.route('/conversations')
def conversations():
    """View all conversations with customers"""
    try:
        page = request.args.get('page', 1, type=int)
        customers = Customer.query.order_by(Customer.last_interaction.desc()).paginate(
            page=page, per_page=20, error_out=False
        )
        
        return render_template('conversations.html', customers=customers)
    except Exception as e:
        logger.error(f"Error loading conversations: {e}")
        return render_template('conversations.html', error=str(e))

@app.route('/customer/<int:customer_id>')
def customer_detail(customer_id):
    """View detailed conversation history for a customer"""
    try:
        customer = Customer.query.get_or_404(customer_id)
        conversations = Conversation.query.filter_by(
            customer_id=customer_id
        ).order_by(Conversation.timestamp.asc()).all()
        
        return render_template('customer_detail.html', 
                             customer=customer, 
                             conversations=conversations)
    except Exception as e:
        logger.error(f"Error loading customer detail: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics')
def analytics():
    """Analytics dashboard with AI learning insights"""
    try:
        # Get learning statistics
        learning_stats = learner.get_learning_statistics()
        
        # Sales over time (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        daily_sales = db.session.query(
            db.func.date(Sale.sale_date),
            db.func.count(Sale.id),
            db.func.sum(Sale.sale_amount)
        ).filter(Sale.sale_date >= thirty_days_ago).group_by(
            db.func.date(Sale.sale_date)
        ).all()
        
        # Customer sentiment analysis
        sentiment_data = db.session.query(
            db.func.avg(Conversation.sentiment_score)
        ).filter(Conversation.sentiment_score.isnot(None)).scalar() or 0
        
        return render_template('analytics.html',
                             learning_stats=learning_stats,
                             daily_sales=daily_sales,
                             avg_sentiment=sentiment_data)
    except Exception as e:
        logger.error(f"Error loading analytics: {e}")
        return render_template('analytics.html', error=str(e))

@app.route('/webhook', methods=['GET', 'POST'])
def web_chat_webhook(): # Renomeado para evitar conflito com a lógica original
    """Handle incoming messages from the web chat interface"""
    
    if request.method == 'GET':
        # Webhook verification, retornando 200 OK para o browser
        return 'OK', 200
    
    elif request.method == 'POST':
        try:
            webhook_data = request.get_json()
            message_content = webhook_data.get('message', '')
            sender_id = webhook_data.get('sender', 'web_user')
            
            if not message_content:
                return jsonify([{"text": "Mensagem vazia."}]), 400

            # Processa a mensagem e obtém a resposta da IA diretamente
            ai_response = process_incoming_message_for_web(sender_id, message_content)
            
            # Retorna a resposta da IA no formato esperado pelo JavaScript
            return jsonify([{"recipient_id": sender_id, "text": ai_response}]), 200
            
        except Exception as e:
            logger.error(f"Error processing webhook from web chat: {e}")
            return jsonify([{"text": "Desculpe, ocorreu um erro no servidor. Tente novamente."}]), 500

def process_incoming_message_for_web(sender_id, message_content):
    """Generate AI response for web chat messages"""
    try:
        # Garante que os handlers foram inicializados
        global ai_agent, learner
        if ai_agent is None or learner is None:
            init_handlers()
        
        # Get or create customer
        customer = Customer.query.filter_by(whatsapp_number=sender_id).first()
        if not customer:
            customer = Customer(
                whatsapp_number=sender_id,
                name="Web Chat User",
                first_contact=datetime.utcnow(),
                last_interaction=datetime.utcnow(),
                total_interactions=0
            )
            db.session.add(customer)
            db.session.flush()
        else:
            customer.last_interaction = datetime.utcnow()
        
        customer.total_interactions += 1
        
        # Get conversation history
        conversation_history = Conversation.query.filter_by(
            customer_id=customer.id
        ).order_by(Conversation.timestamp.asc()).limit(20).all()
        
        conversation_dict = []
        for conv in conversation_history:
            conversation_dict.append({
                'message_type': conv.message_type,
                'message_content': conv.message_content
            })
        
        # Generate AI response
        ai_strategy = learner.get_best_strategy({}) # Pass a default analysis for web chat
        ai_response = ai_agent.generate_response(
            message_content, {}, conversation_dict, ai_strategy
        )
        
        # Save both messages (user and AI)
        incoming_conversation = Conversation(
            customer_id=customer.id,
            message_type='incoming',
            message_content=message_content,
            timestamp=datetime.utcnow()
        )
        outgoing_conversation = Conversation(
            customer_id=customer.id,
            message_type='outgoing',
            message_content=ai_response,
            timestamp=datetime.utcnow(),
            ai_strategy=ai_strategy
        )
        db.session.add(incoming_conversation)
        db.session.add(outgoing_conversation)
        
        db.session.commit()
        logger.info(f"Successfully processed web chat message from {sender_id}")
        return ai_response
            
    except Exception as e:
        logger.error(f"Error processing web chat message from {sender_id}: {e}")
        db.session.rollback()
        # Retorna uma mensagem de erro genérica para o frontend
        return "Desculpe, ocorreu um erro no servidor. Tente novamente."


# As rotas a seguir são mantidas inalteradas

@app.route('/simulate_sale', methods=['POST'])
def simulate_sale():
    # ... (código existente) ...
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({"error": "customer_id required"}), 400
        
        customer = Customer.query.get(customer_id)
        if not customer:
            return jsonify({"error": "Customer not found"}), 404
        
        # Get the latest conversation to determine strategy used
        latest_conv = Conversation.query.filter_by(
            customer_id=customer_id,
            message_type='outgoing'
        ).order_by(Conversation.timestamp.desc()).first()
        
        strategy_used = latest_conv.ai_strategy if latest_conv else "consultivo"
        
        # Create sale record
        sale = Sale(
            customer_id=customer_id,
            product_name="Curso Digital de Marketing",
            sale_amount=297.00,
            sale_date=datetime.utcnow(),
            conversation_messages=customer.total_interactions
        )
        db.session.add(sale)
        
        # Update customer
        customer.purchased = True
        customer.purchase_date = datetime.utcnow()
        
        # Record success in reinforcement learning
        learner.record_success(customer_id, strategy_used, customer.total_interactions)
        
        db.session.commit()
        
        return jsonify({"message": "Sale recorded successfully", "strategy": strategy_used})
        
    except Exception as e:
        logger.error(f"Error simulating sale: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/learning_stats')
def api_learning_stats():
    # ... (código existente) ...
    try:
        stats = learner.get_learning_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/products')
def products():
    # ... (código existente) ...
    try:
        products = Product.query.filter_by(is_active=True).order_by(Product.created_at.desc()).all()
        return render_template('products.html', products=products)
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        return render_template('products.html', error=str(e))

@app.route('/products/new', methods=['GET', 'POST'])
def new_product():
    # ... (código existente) ...
    if request.method == 'POST':
        try:
            import json
            
            # Get form data
            name = request.form.get('name')
            niche = request.form.get('niche')
            price = float(request.form.get('price', 0))
            description = request.form.get('description')
            target_audience = request.form.get('target_audience')
            sales_approach = request.form.get('sales_approach', 'consultivo')
            
            # Process benefits - convert from textarea to JSON
            benefits_text = request.form.get('key_benefits', '')
            benefits_list = [benefit.strip() for benefit in benefits_text.split('\n') if benefit.strip()]
            key_benefits = json.dumps(benefits_list)
            
            # Create new product
            product = Product(
                name=name,
                niche=niche,
                price=price,
                description=description,
                target_audience=target_audience,
                key_benefits=key_benefits,
                sales_approach=sales_approach
            )
            
            db.session.add(product)
            db.session.commit()
            
            flash(f'Produto "{name}" criado com sucesso!', 'success')
            return redirect(url_for('products'))
            
        except Exception as e:
            logger.error(f"Error creating product: {e}")
            flash(f'Erro ao criar produto: {str(e)}', 'error')
            db.session.rollback()
    
    return render_template('new_product.html')

@app.route('/products/<int:product_id>/edit', methods=['GET', 'POST'])
def edit_product(product_id):
    # ... (código existente) ...
    product = Product.query.get_or_404(product_id)
    
    if request.method == 'POST':
        try:
            import json
            
            # Update product data
            product.name = request.form.get('name')
            product.niche = request.form.get('niche')
            product.price = float(request.form.get('price', 0))
            product.description = request.form.get('description')
            product.target_audience = request.form.get('target_audience')
            product.sales_approach = request.form.get('sales_approach', 'consultivo')
            
            # Process benefits
            benefits_text = request.form.get('key_benefits', '')
            benefits_list = [benefit.strip() for benefit in benefits_text.split('\n') if benefit.strip()]
            product.key_benefits = json.dumps(benefits_list)
            
            product.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            flash(f'Produto "{product.name}" atualizado com sucesso!', 'success')
            return redirect(url_for('products'))
            
        except Exception as e:
            logger.error(f"Error updating product: {e}")
            flash(f'Erro ao atualizar produto: {str(e)}', 'error')
            db.session.rollback()
    
    # Convert benefits JSON back to text for form
    import json
    try:
        benefits_list = json.loads(product.key_benefits)
        benefits_text = '\n'.join(benefits_list)
    except:
        benefits_text = product.key_benefits
    
    return render_template('edit_product.html', product=product, benefits_text=benefits_text)

@app.route('/products/<int:product_id>/delete', methods=['POST'])
def delete_product(product_id):
    # ... (código existente) ...
    try:
        product = Product.query.get_or_404(product_id)
        product.is_active = False
        product.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        flash(f'Produto "{product.name}" removido com sucesso!', 'success')
        return redirect(url_for('products'))
        
    except Exception as e:
        logger.error(f"Error deleting product: {e}")
        flash(f'Erro ao remover produto: {str(e)}', 'error')
        return redirect(url_for('products'))

@app.route('/niches')
def niches():
    # ... (código existente) ...
    try:
        # Get niche statistics
        from sqlalchemy import func
        
        niche_stats = db.session.query(
            Product.niche,
            func.count(Product.id).label('total_products'),
            func.count(Sale.id).label('total_sales'),
            func.sum(Sale.sale_amount).label('total_revenue'),
            func.avg(Sale.sale_amount).label('avg_sale_value')
        ).outerjoin(Sale, Product.id == Sale.product_id
        ).filter(Product.is_active == True
        ).group_by(Product.niche).all()
        
        return render_template('niches.html', niche_stats=niche_stats)
        
    except Exception as e:
        logger.error(f"Error loading niches: {e}")
        return render_template('niches.html', error=str(e))
