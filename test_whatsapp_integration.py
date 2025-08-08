#!/usr/bin/env python3
"""
WhatsApp Integration Test Script
Tests the basic connectivity and configuration of your WhatsApp Business API setup.
"""

import os
import requests
import json
from datetime import datetime

def test_environment_variables():
    """Test if all required environment variables are set"""
    print("🔍 Checking Environment Variables...")
    
    required_vars = [
        'OPENROUTER_API_KEY',
        'WHATSAPP_ACCESS_TOKEN',
        'WHATSAPP_PHONE_NUMBER_ID',
        'WHATSAPP_VERIFY_TOKEN'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var}: Set")
    
    if missing_vars:
        print(f"❌ Missing variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ All environment variables are set!")
    return True

def test_openrouter_api():
    """Test OpenRouter API connectivity"""
    print("\n🤖 Testing OpenRouter API...")
    
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        return False
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'meta-llama/llama-3.1-8b-instruct:free',
            'messages': [
                {'role': 'user', 'content': 'Responda apenas "OK" se você conseguir me entender em português.'}
            ],
            'max_tokens': 10
        }
        
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            print(f"✅ OpenRouter API working: {ai_response.strip()}")
            return True
        else:
            print(f"❌ OpenRouter API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ OpenRouter API error: {e}")
        return False

def test_whatsapp_api():
    """Test WhatsApp Business API connectivity"""
    print("\n📱 Testing WhatsApp Business API...")
    
    access_token = os.environ.get('WHATSAPP_ACCESS_TOKEN')
    phone_number_id = os.environ.get('WHATSAPP_PHONE_NUMBER_ID')
    
    if not access_token or not phone_number_id:
        print("❌ WhatsApp credentials not set")
        return False
    
    try:
        # Test API connection by getting phone number info
        url = f"https://graph.facebook.com/v17.0/{phone_number_id}"
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            phone_info = response.json()
            print(f"✅ WhatsApp API connected")
            print(f"   Phone Number: +{phone_info.get('display_phone_number', 'N/A')}")
            print(f"   Status: {phone_info.get('verified_name', 'N/A')}")
            return True
        else:
            print(f"❌ WhatsApp API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ WhatsApp API error: {e}")
        return False

def test_webhook_url():
    """Test if webhook URL is accessible"""
    print("\n🔗 Testing Webhook URL...")
    
    # Get the current Replit domain
    replit_domain = os.environ.get('REPLIT_DOMAIN')
    if replit_domain:
        webhook_url = f"https://{replit_domain}/webhook"
    else:
        print("⚠️  REPLIT_DOMAIN not found, using localhost")
        webhook_url = "http://localhost:5000/webhook"
    
    try:
        # Test GET request to webhook (verification endpoint)
        verify_token = os.environ.get('WHATSAPP_VERIFY_TOKEN', 'test_token')
        params = {
            'hub.mode': 'subscribe',
            'hub.verify_token': verify_token,
            'hub.challenge': 'test_challenge'
        }
        
        response = requests.get(webhook_url, params=params, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Webhook URL accessible: {webhook_url}")
            print(f"   Response: {response.text}")
            return True
        else:
            print(f"❌ Webhook error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return False

def test_database_connection():
    """Test database connectivity"""
    print("\n💾 Testing Database Connection...")
    
    try:
        # Import our models to test database
        import sys
        sys.path.append('.')
        
        from app import app, db
        from models import Customer, Product, Conversation
        
        with app.app_context():
            # Test basic database queries
            customer_count = Customer.query.count()
            product_count = Product.query.count()
            conversation_count = Conversation.query.count()
            
            print(f"✅ Database connected successfully")
            print(f"   Customers: {customer_count}")
            print(f"   Products: {product_count}")
            print(f"   Conversations: {conversation_count}")
            
            return True
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 WhatsApp AI Sales Agent - Integration Test")
    print("=" * 50)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("OpenRouter API", test_openrouter_api),
        ("WhatsApp Business API", test_whatsapp_api),
        ("Webhook URL", test_webhook_url),
        ("Database Connection", test_database_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your WhatsApp AI Sales Agent is ready to go!")
        print("\n📝 Next Steps:")
        print("1. Deploy your app using the Deploy button")
        print("2. Configure WhatsApp webhook with your deployed URL")
        print("3. Add your first products in the Products section")
        print("4. Start receiving customer messages!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above before deploying.")
        print("\n🔧 Common Solutions:")
        print("- Add missing environment variables in Replit Secrets")
        print("- Verify API keys are correct and active")
        print("- Check WhatsApp Business API setup")

if __name__ == "__main__":
    main()