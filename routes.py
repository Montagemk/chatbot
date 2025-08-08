from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app, db
from models import Customer, Conversation, Sale, WhatsAppWebhook, AILearningData, Product
from whatsapp_handler import WhatsAppHandler
from ai_agent import AIAgent
from reinforcement_learning import ReinforcementLearner
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

# Initialize handlers - these will be initialized when app starts
whatsapp = None
ai_agent = None
learner = None

def init_handlers():
    """Initialize handlers within application context"""
    global whatsapp, ai_agent, learner
    whatsapp = WhatsAppHandler()
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
def whatsapp_webhook():
    """Handle WhatsApp webhook verification and incoming messages"""
    
    if request.method == 'GET':
        # Webhook verification
        mode = request.args.get('hub.mode', '')
        token = request.args.get('hub.verify_token', '')
        challenge = request.args.get('hub.challenge', '')
        
        verification_result = whatsapp.verify_webhook(mode, token, challenge)
        if verification_result:
            return verification_result
        else:
            return 'Verification failed', 403
    
    elif request.method == 'POST':
        # Handle incoming messages
        try:
            webhook_data = request.get_json()
            
            # Store webhook data for debugging
            webhook_record = WhatsAppWebhook(
                webhook_data=json.dumps(webhook_data),
                processed=False
            )
            db.session.add(webhook_record)
            db.session.commit()
            
            # Parse message data
            message_data = whatsapp.parse_webhook_data(webhook_data)
            if not message_data:
                webhook_record.processed = True
                webhook_record.error_message = "No valid message data found"
                db.session.commit()
                return 'OK', 200
            
            # Process the message
            success = process_incoming_message(message_data)
            
            webhook_record.processed = True
            if not success:
                webhook_record.error_message = "Failed to process message"
            
            db.session.commit()
            
            return 'OK', 200
            
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return 'Error', 500

def process_incoming_message(message_data):
    """Process incoming WhatsApp message and generate AI response"""
    try:
        phone_number = message_data['from']
        message_content = message_data['text']
        message_id = message_data['message_id']
        profile_name = message_data.get('profile_name', '')
        
        # Mark message as read
        whatsapp.mark_as_read(message_id)
        
        # Get or create customer
        customer = Customer.query.filter_by(whatsapp_number=phone_number).first()
        if not customer:
            customer = Customer(
                whatsapp_number=phone_number,
                name=profile_name,
                first_contact=datetime.utcnow(),
                last_interaction=datetime.utcnow(),
                total_interactions=0
            )
            db.session.add(customer)
            db.session.flush()
        else:
            customer.last_interaction = datetime.utcnow()
        
        customer.total_interactions += 1
        
        # Save incoming message
        incoming_conversation = Conversation(
            customer_id=customer.id,
            message_type='incoming',
            message_content=message_content,
            timestamp=datetime.utcnow()
        )
        db.session.add(incoming_conversation)
        db.session.flush()
        
        # Get conversation history
        conversation_history = Conversation.query.filter_by(
            customer_id=customer.id
        ).order_by(Conversation.timestamp.asc()).limit(20).all()
        
        conversation_dict = []
        for conv in conversation_history:
            conversation_dict.append({
                'message_type': conv.message_type,
                'message_content': conv.message_content,
                'timestamp': conv.timestamp.isoformat(),
                'ai_strategy': conv.ai_strategy
            })
        
        # Analyze customer intent and sentiment
        customer_analysis = ai_agent.analyze_customer_intent(message_content, conversation_dict)
        
        # Update sentiment in the conversation record
        incoming_conversation.sentiment_score = customer_analysis.get('sentiment', 0.0)
        
        # Check if customer is ready to buy
        purchase_intent = ai_agent.detect_purchase_intent(message_content, conversation_dict)
        
        # Generate AI response
        ai_strategy = learner.get_best_strategy(customer_analysis)
        ai_response = ai_agent.generate_response(
            message_content, customer_analysis, conversation_dict, ai_strategy
        )
        
        # Handle purchase intent
        if purchase_intent.get('ready_to_buy', False) and purchase_intent.get('confidence', 0) > 0.7:
            # Add payment information to response
            payment_link = f"Para finalizar sua compra, acesse: https://exemplo.com/checkout/{customer.id}"
            ai_response += f"\n\n💳 {payment_link}"
            
            # This would typically integrate with a payment processor
            # For now, we'll simulate sale detection with a webhook or manual confirmation
        
        # Save AI response
        outgoing_conversation = Conversation(
            customer_id=customer.id,
            message_type='outgoing',
            message_content=ai_response,
            timestamp=datetime.utcnow(),
            ai_strategy=ai_strategy
        )
        db.session.add(outgoing_conversation)
        
        # Send response via WhatsApp
        success = whatsapp.send_message(phone_number, ai_response)
        
        if success:
            db.session.commit()
            logger.info(f"Successfully processed message from {phone_number}")
            return True
        else:
            db.session.rollback()
            logger.error(f"Failed to send WhatsApp message to {phone_number}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing incoming message: {e}")
        db.session.rollback()
        return False

@app.route('/simulate_sale', methods=['POST'])
def simulate_sale():
    """Simulate a sale for testing reinforcement learning"""
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
    """API endpoint for learning statistics"""
    try:
        stats = learner.get_learning_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/products')
def products():
    """Product management page"""
    try:
        products = Product.query.filter_by(is_active=True).order_by(Product.created_at.desc()).all()
        return render_template('products.html', products=products)
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        return render_template('products.html', error=str(e))

@app.route('/products/new', methods=['GET', 'POST'])
def new_product():
    """Create new product"""
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
    """Edit existing product"""
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
    """Delete product (mark as inactive)"""
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
    """View all product niches and their performance"""
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
