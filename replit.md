# WhatsApp AI Sales Agent

## Overview

This is an intelligent WhatsApp sales agent that uses AI and reinforcement learning to autonomously conduct sales conversations. The system integrates with WhatsApp Business API to interact with customers in Portuguese, using OpenAI's GPT-4o to generate dynamic, contextual responses. The AI learns from each interaction, optimizing sales strategies based on successful conversions to maximize revenue from digital product sales.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Web Framework Architecture
- **Flask-based web application** with SQLAlchemy ORM for database operations
- **Blueprint-style routing** centralized in routes.py for clean separation of concerns
- **Template-based frontend** using Jinja2 templates with Bootstrap for responsive UI
- **RESTful API endpoints** for webhook handling and data retrieval

### AI Agent Architecture
- **OpenRouter API integration** with Meta Llama 3.1 8B Instruct model for natural language generation and customer intent analysis
- **Intent classification system** that categorizes customer messages into predefined intents (initial interest, questions, objections, ready to buy, disinterested)
- **Sentiment analysis** with numerical scoring from -1 to 1 for customer mood tracking
- **Dynamic response generation** based on customer context, conversation history, and product information

### Reinforcement Learning System
- **Multi-armed bandit approach** with four sales strategies: consultivo (consultative), escassez (scarcity), emocional (emotional), racional (rational)
- **Exploration vs exploitation balance** with 20% exploration rate and 95% decay factor
- **Success tracking** based on actual sales conversions as reward signals
- **Strategy optimization** that learns which approaches work best for different customer types and contexts

### Database Schema
- **Customer table** tracking WhatsApp numbers, interaction history, and purchase status
- **Conversation table** storing all messages with AI strategy metadata and sentiment scores
- **Sale table** recording successful conversions with revenue tracking
- **AILearningData table** maintaining strategy performance metrics and success rates

### WhatsApp Integration
- **Webhook verification** for secure Facebook/WhatsApp API integration
- **Message parsing** to extract customer information and message content
- **Bidirectional messaging** supporting both incoming message processing and outgoing response delivery
- **Phone number validation** and customer identification system

### Analytics and Monitoring
- **Real-time dashboard** displaying conversion rates, revenue metrics, and customer statistics
- **Strategy performance tracking** showing success rates for different AI approaches
- **Conversation history viewer** for manual review and quality assurance
- **Learning statistics** to monitor AI improvement over time

## External Dependencies

### AI Services
- **OpenRouter API** for natural language processing and response generation using Meta Llama 3.1 8B Instruct (free model)
- Requires OPENROUTER_API_KEY environment variable
- Provides cost-effective access to multiple AI models with generous free tier

### Communication Platform
- **WhatsApp Business API** via Facebook Graph API v17.0
- Requires WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID, and WHATSAPP_VERIFY_TOKEN

### Database
- **SQLAlchemy-compatible database** (defaults to SQLite, configurable via DATABASE_URL)
- Supports PostgreSQL, MySQL, and other SQL databases through connection string configuration

### Frontend Dependencies
- **Bootstrap 5** with dark theme support for responsive UI components
- **Font Awesome 6.4.0** for iconography
- **Chart.js** for data visualization and analytics charts

### Python Libraries
- **Flask** web framework with SQLAlchemy for database operations
- **Requests** library for HTTP API calls to OpenRouter and WhatsApp APIs
- **Werkzeug ProxyFix** for proper header handling in production deployments

### Configuration
- **Environment-based configuration** for API keys, database connections, and product information
- **Flexible product configuration** supporting name, price, description, and benefits customization
- **Session management** with configurable secret keys for security

## Deployment and Integration

### WhatsApp Business API Setup
- Requires Facebook Developer account and WhatsApp Business API approval
- Webhook integration at `/webhook` endpoint for real-time message processing
- Environment variables needed: `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`

### Deployment Process
- Ready for Replit Autoscale deployment for automatic scaling
- Production-ready with proper error handling and logging
- Database automatically created on first run
- Includes test script (`test_whatsapp_integration.py`) for verifying setup

### Recent Changes (August 2024)
- Migrated from OpenAI to OpenRouter API for cost optimization
- Added comprehensive product management system with niche support
- Created specialized sales strategy selection (consultivo, escassez, emocional, racional)
- Built performance analytics dashboard for tracking niche effectiveness
- Integrated Brazilian Portuguese language optimization for local market