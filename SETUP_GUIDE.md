# WhatsApp AI Sales Agent - Setup Guide

## 1. Deploy Your Application on Replit

### Current Status
✅ Your application is already running on Replit
✅ OpenRouter API key is configured
✅ Database is set up and working

### Deploy to Production
1. **Click the "Deploy" button** in the top-right corner of your Replit workspace
2. **Choose "Autoscale"** for automatic scaling based on traffic
3. **Set your domain name** (optional) or use the provided `.replit.app` domain
4. **Configure environment variables** in the deployment settings:
   - `OPENROUTER_API_KEY` (already set)
   - `WHATSAPP_ACCESS_TOKEN` (you'll get this from Facebook)
   - `WHATSAPP_PHONE_NUMBER_ID` (you'll get this from Facebook)
   - `WHATSAPP_VERIFY_TOKEN` (create a random string, e.g., "minha_verificacao_123")

## 2. Set Up WhatsApp Business API

### Step 1: Create Facebook Developer Account
1. Go to https://developers.facebook.com/
2. Create a developer account if you don't have one
3. Complete identity verification (may take 1-2 days)

### Step 2: Create a Meta App
1. Click "Create App" → "Business" → "WhatsApp"
2. Fill in your app details:
   - **App Name**: "IA Vendas [Seu Nome]"
   - **App Contact Email**: Your email
   - **Business Account**: Create or select one

### Step 3: Configure WhatsApp Business API
1. In your app dashboard, go to "WhatsApp" → "API Setup"
2. **Add Phone Number**:
   - Add your business phone number
   - Verify with SMS/call
   - Accept WhatsApp Business terms

3. **Get Your Credentials**:
   - **Phone Number ID**: Copy this number
   - **Access Token**: Generate a permanent token
   - **Webhook Verify Token**: Create a secure random string

### Step 4: Configure Webhook
1. In WhatsApp API Setup, go to "Webhook"
2. **Webhook URL**: `https://your-replit-app.replit.app/webhook`
3. **Verify Token**: Use the same token you created above
4. **Subscribe to Fields**: Check "messages"

## 3. Configure Your Replit App

### Add Environment Variables
Add these to your Replit Secrets:

```bash
WHATSAPP_ACCESS_TOKEN=your_permanent_access_token_from_facebook
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id_from_facebook
WHATSAPP_VERIFY_TOKEN=your_custom_verification_token
```

### Test Webhook Connection
1. After adding the secrets, restart your Replit app
2. Facebook will verify your webhook automatically
3. Check the console logs for successful verification

## 4. Test Your AI Sales Agent

### Send Test Message
1. Send a WhatsApp message to your business number
2. Check the Replit console for incoming webhook data
3. Verify the AI responds in Portuguese with sales approach

### Monitor Dashboard
1. Visit your deployed app's dashboard
2. Check conversation tracking
3. Monitor AI learning statistics

## 5. Production Checklist

### Security
- [ ] Use strong verification tokens
- [ ] Never share access tokens publicly
- [ ] Enable webhook signature verification (optional but recommended)

### Compliance
- [ ] Add WhatsApp Business terms to your website
- [ ] Include privacy policy mentioning AI usage
- [ ] Follow WhatsApp Business messaging guidelines
- [ ] Respect opt-out requests

### Monitoring
- [ ] Set up alerts for failed webhook deliveries
- [ ] Monitor API rate limits
- [ ] Track conversion metrics
- [ ] Backup conversation data regularly

## 6. Advanced Features

### Payment Integration
- Configure payment links in the AI responses
- Integrate with Brazilian payment processors (PagSeguro, Mercado Pago)
- Set up webhook for purchase confirmations

### Multi-Product Support
- Use the product management interface to add your digital products
- Define specific niches and target audiences
- Let the AI learn optimal strategies for each product

### Analytics
- Monitor niche performance
- Track conversion rates by strategy
- Optimize based on AI learning data

## Troubleshooting

### Common Issues
1. **Webhook not receiving messages**:
   - Check webhook URL is publicly accessible
   - Verify environment variables are set correctly
   - Ensure phone number is verified

2. **AI not responding**:
   - Check OpenRouter API key is valid
   - Monitor console logs for errors
   - Verify Portuguese language model is working

3. **Database errors**:
   - Check if all tables were created successfully
   - Restart the application to recreate tables
   - Monitor SQLite file permissions

### Support Resources
- Facebook Developer Documentation: https://developers.facebook.com/docs/whatsapp
- WhatsApp Business API: https://business.whatsapp.com/developers
- OpenRouter Documentation: https://openrouter.ai/docs

---

**Important**: This setup connects your app to real customers. Test thoroughly with a small group before launching to your full audience.