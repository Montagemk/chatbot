import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        # OpenRouter API setup
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "demo_key")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3.1-8b-instruct:free"  # Free model on OpenRouter
        
        # O HTTP-Referer deve ser o endereço do seu servidor no Render
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("RENDER_EXTERNAL_URL", "https://atpchatbot.onrender.com"),
            "X-Title": "WhatsApp AI Sales Agent"
        }
        
        # Product information (can be configured via environment or database)
        self.product_info = {
            "name": os.environ.get("PRODUCT_NAME", "Curso Digital de Marketing"),
            "price": float(os.environ.get("PRODUCT_PRICE", "297.00")),
            "description": os.environ.get("PRODUCT_DESCRIPTION", "Aprenda as estratégias mais eficazes de marketing digital"),
            "benefits": [
                "Aumento de vendas em até 300%",
                "Suporte vitalício",
                "Certificado reconhecido",
                "Acesso a comunidade exclusiva"
            ]
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 500) -> Optional[Dict[str, Any]]:
        """Make API call to OpenRouter"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error making OpenRouter API call: {e}")
            return None
    
    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze customer intent and sentiment from message"""
        try:
            # Build context from conversation history
            context = self._build_conversation_context(conversation_history)
            
            prompt = f"""
            Você é um especialista em análise de intenção de clientes brasileiros. 
            Analise a mensagem do cliente e determine:
            
            Contexto da conversa: {context}
            Mensagem atual: {message}
            
            Responda APENAS em formato JSON válido com:
            {{
                "intent": "interesse_inicial",
                "sentiment": 0.0,
                "urgency": 0.5,
                "keywords": ["palavra1", "palavra2"],
                "objections": [],
                "buying_signals": []
            }}
            
            Valores para intent: interesse_inicial, duvidas, objecoes, pronto_comprar, desinteressado
            Valores para sentiment: número de -1 a 1 (negativo a positivo)
            Valores para urgency: número de 0 a 1 (baixa a alta urgência)
            """
            
            response = self._make_api_call([
                {"role": "system", "content": "Você é um especialista em análise de clientes brasileiros. Responda APENAS com JSON válido."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                if content:
                    result = json.loads(content)
                    logger.info(f"Customer intent analysis: {result}")
                    return result
            
        except Exception as e:
            logger.error(f"Error analyzing customer intent: {e}")
            
        return {
            "intent": "interesse_inicial",
            "sentiment": 0.0,
            "urgency": 0.5,
            "keywords": [],
            "objections": [],
            "buying_signals": []
        }
    
    def generate_response(self, customer_message: str, customer_analysis: Dict, 
                         conversation_history: List[Dict], strategy: str = "adaptive") -> str:
        """Generate AI response using reinforcement learning insights"""
        try:
            # Get the best performing strategy from learning data
            if strategy == "adaptive":
                from reinforcement_learning import ReinforcementLearner
                learner = ReinforcementLearner()
                strategy = learner.get_best_strategy(customer_analysis)
            
            # Build conversation context
            context = self._build_conversation_context(conversation_history, limit=5)
            
            # Create system prompt based on strategy and Brazilian culture
            system_prompt = self._create_system_prompt(strategy)
            
            # Create user prompt with all context
            user_prompt = f"""
            ANÁLISE DO CLIENTE:
            - Intenção: {customer_analysis.get('intent', 'interesse_inicial')}
            - Sentimento: {customer_analysis.get('sentiment', 0)}
            - Urgência: {customer_analysis.get('urgency', 0.5)}
            - Objeções: {customer_analysis.get('objections', [])}
            - Sinais de compra: {customer_analysis.get('buying_signals', [])}
            
            CONTEXTO DA CONVERSA:
            {context}
            
            MENSAGEM ATUAL DO CLIENTE:
            {customer_message}
            
            INFORMAÇÕES DO PRODUTO:
            - Nome: {self.product_info['name']}
            - Preço: R$ {self.product_info['price']:.2f}
            - Descrição: {self.product_info['description']}
            - Benefícios: {', '.join(self.product_info['benefits'])}
            
            Gere uma resposta persuasiva, natural e culturalmente adequada para brasileiros.
            Use técnicas de copywriting e seja empático. Mantenha tom conversacional.
            """
            
            response = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.7, max_tokens=300)
            
            if response and 'choices' in response and response['choices'][0]['message']['content']:
                content = response['choices'][0]['message']['content']
                if content:
                    ai_response = content.strip()
                    logger.info(f"Generated AI response using strategy: {strategy}")
                    return ai_response
            
            # Se a resposta for vazia ou inválida, retorna uma mensagem de erro
            logger.error("OpenRouter API returned an invalid or empty response.")
            return "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
    
    def _build_conversation_context(self, conversation_history: List[Dict], limit: int = 10) -> str:
        """Build conversation context string from history"""
        if not conversation_history:
            return "Primeira interação com o cliente."
        
        # Get last N messages
        recent_messages = conversation_history[-limit:] if len(conversation_history) > limit else conversation_history
        
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Você"
            context_lines.append(f"{msg_type}: {msg.get('message_content', '')}")
        
        return "\n".join(context_lines)
    
    def _create_system_prompt(self, strategy: str) -> str:
        """Create system prompt based on strategy and Brazilian cultural context"""
        base_prompt = """
        Você é um vendedor brasileiro especialista em produtos digitais. Suas características:
        
        PERSONALIDADE:
        - Caloroso e amigável, típico brasileiro
        - Confiável e profissional
        - Empático e bom ouvinte
        - Persuasivo sem ser agressivo
        
        LINGUAGEM:
        - Use português brasileiro natural
        - Inclua expressões brasileiras quando apropriado
        - Tom conversacional e próximo
        - Evite linguagem muito formal
        
        OBJETIVO:
        - Vender o produto digital oferecido
        - Construir relacionamento e confiança
        - Superar objeções de forma inteligente
        - Criar urgência quando apropriado
        """
        
        strategy_prompts = {
            "consultivo": """
            ESTRATÉGIA CONSULTIVA:
            - Faça perguntas para entender as necessidades do cliente
            - Posicione-se como consultor, não apenas vendedor
            - Mostre como o produto resolve problemas específicos
            - Use histórias de sucesso de outros clientes brasileiros
            """,
            
            "escassez": """
            ESTRATÉGIA DE ESCASSEZ:
            - Mencione vagas limitadas ou promoção por tempo limitado
            - Crie senso de urgência respeitoso
            - Use prova social (outros brasileiros comprando)
            - Destaque benefícios únicos e exclusivos
            """,
            
            "emocional": """
            ESTRATÉGIA EMOCIONAL:
            - Conecte-se com sonhos e aspirações do cliente
            - Use storytelling envolvente
            - Destaque transformação de vida que o produto proporciona
            - Seja empático com desafios financeiros brasileiros
            """,
            
            "racional": """
            ESTRATÉGIA RACIONAL:
            - Apresente dados e resultados concretos
            - Compare custo-benefício de forma clara
            - Mencione garantias e suporte
            - Use argumentos lógicos e bem estruturados
            """
        }
        
        strategy_prompt = strategy_prompts.get(strategy, strategy_prompts["consultivo"])
        
        return base_prompt + "\n" + strategy_prompt + """
        
        IMPORTANTE:
        - Sempre responda em português brasileiro
        - Mantenha mensagens concisas (máximo 2-3 parágrafos)
        - Termine com uma pergunta ou call-to-action quando apropriado
        - Adapte-se ao nível de interesse demonstrado pelo cliente
        """

    def detect_purchase_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Detect if customer is ready to purchase"""
        try:
            context = self._build_conversation_context(conversation_history)
            
            prompt = f"""
            Analise se o cliente está pronto para comprar com base na conversa.
            
            Contexto: {context}
            Mensagem atual: {message}
            
            Responda em JSON:
            {{
                "ready_to_buy": true/false,
                "confidence": número de 0 a 1,
                "purchase_signals": ["sinal1", "sinal2"],
                "next_action": "send_payment_link|continue_conversation|address_objections"
            }}
            """
            
            response = self._make_api_call([
                {"role": "system", "content": "Você é um especialista em detectar intenção de compra. Responda APENAS com JSON válido."},
                {"role": "user", "content": prompt}
            ], temperature=0.2)
            
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                if content:
                    return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error detecting purchase intent: {e}")
            return {
                "ready_to_buy": False,
                "confidence": 0.0,
                "purchase_signals": [],
                "next_action": "continue_conversation"
            }
}
