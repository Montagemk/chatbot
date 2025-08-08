import os
import json
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WhatsAppHandler:
    def __init__(self):
        # WhatsApp Business API credentials from environment
        self.access_token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "demo_token")
        self.phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "demo_phone_id")
        self.verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "whatsapp_verify_token_2024")
        self.base_url = "https://graph.facebook.com/v17.0"
        
    def verify_webhook(self, mode: str, token: str, challenge: str) -> Optional[str]:
        """Verify WhatsApp webhook during setup"""
        if mode == "subscribe" and token == self.verify_token:
            logger.info("WhatsApp webhook verified successfully")
            return challenge
        logger.warning(f"WhatsApp webhook verification failed: mode={mode}, token={token}")
        return None
    
    def parse_webhook_data(self, webhook_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse incoming WhatsApp webhook data"""
        try:
            if not webhook_data.get("entry"):
                return None
                
            entry = webhook_data["entry"][0]
            if not entry.get("changes"):
                return None
                
            change = entry["changes"][0]
            if change.get("field") != "messages":
                return None
                
            value = change.get("value", {})
            messages = value.get("messages", [])
            
            if not messages:
                return None
                
            message = messages[0]
            
            return {
                "from": message.get("from"),
                "message_id": message.get("id"),
                "timestamp": message.get("timestamp"),
                "message_type": message.get("type", "text"),
                "text": message.get("text", {}).get("body", "") if message.get("type") == "text" else "",
                "profile_name": value.get("contacts", [{}])[0].get("profile", {}).get("name", "")
            }
            
        except Exception as e:
            logger.error(f"Error parsing WhatsApp webhook data: {e}")
            return None
    
    def send_message(self, to: str, message: str, message_type: str = "text") -> bool:
        """Send message via WhatsApp Business API"""
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": message_type,
            "text": {"body": message}
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {to}")
                return True
            else:
                logger.error(f"Failed to send message to {to}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending WhatsApp message to {to}: {e}")
            return False
    
    def mark_as_read(self, message_id: str) -> bool:
        """Mark message as read"""
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error marking message as read: {e}")
            return False
