import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from models import Product # Importa o modelo Product

# Configura o logger para este módulo.
logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        """
        Inicializa o Agente de IA, configurando chaves de API e endpoints.
        """
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "openai/gpt-oss-20b"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("RENDER_EXTERNAL_URL", "https://comunidadeatp.store"),
            "X-Title": "WhatsApp AI Sales Agent"
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 500) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            logger.error("A chave da API OpenRouter não está configurada. Defina a variável de ambiente OPENROUTER_API_KEY.")
            return None
        try:
            payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "timeout": 25}
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                if response.text:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        logger.error(f"Erro na API OpenRouter: Resposta 200 OK, mas o corpo não era um JSON válido. Resposta: {response.text}")
                        return None
                else:
                    logger.error("Erro na API OpenRouter: Resposta 200 OK, mas o corpo da resposta estava vazio.")
                    return None
            else:
                logger.error(f"Erro na API OpenRouter: Recebido status {response.status_code}. Resposta: {response.text}")
                return None
        except requests.exceptions.Timeout:
            logger.error("Erro ao chamar a API OpenRouter: A requisição excedeu o tempo limite (Timeout).")
            return None
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em _make_api_call: {e}")
            return None

    def generate_response(self, customer_message: str, customer_analysis: Dict, 
                         conversation_history: List[Dict], strategy: str = "adaptive", 
                         product: Optional[Product] = None) -> str:
        """
        Gera uma resposta persuasiva usando o contexto completo e atualizado do produto.
        """
        default_error_message = "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        
        # --- LÓGICA DE CONTEXTO DO PRODUTO ATUALIZADA COM TODOS OS NOVOS CAMPOS ---
        if product:
            try:
                benefits_list = json.loads(product.key_benefits)
                benefits_str = ', '.join(benefits_list)
            except (json.JSONDecodeError, TypeError):
                benefits_str = str(product.key_benefits)

            product_info = {
                "name": product.name,
                "price": product.price,
                "original_price": product.original_price,
                "description": product.description,
                "benefits": benefits_str,
                "payment_link": product.payment_link,
                "image_url": product.product_image_url,
                "free_group_link": product.free_group_link
            }
        else:
            product_info = {
                "name": "nosso produto digital", "price": 0, "original_price": None,
                "description": "uma solução completa para suas necessidades.", "benefits": "suporte e mais.",
                "payment_link": None, "image_url": None, "free_group_link": None
            }

        try:
            from reinforcement_learning import ReinforcementLearner
            learner = ReinforcementLearner()
            if strategy == "adaptive":
                strategy = learner.get_best_strategy(customer_analysis)
            
            context = self._build_conversation_context(conversation_history, limit=5)
            system_prompt = self._create_system_prompt(strategy)
            
            # --- PROMPT DO USUÁRIO ATUALIZADO PARA INCLUIR TODAS AS NOVAS INFORMAÇÕES ---
            user_prompt = f"""
            ANÁLISE DO CLIENTE:
            - Intenção: {customer_analysis.get('intent', 'desconhecida')}
            
            CONTEXTO DA CONVERSA:
            {context}
            
            MENSAGEM ATUAL DO CLIENTE:
            {customer_message}
            
            INFORMAÇÕES COMPLETAS DO PRODUTO PARA VENDER:
            - Nome: {product_info['name']}
            - Preço "De": R$ {product_info['original_price']:.2f} (se aplicável, use para mostrar desconto)
            - Preço "Por": R$ {product_info['price']:.2f} (este é o preço final de venda)
            - Descrição: {product_info['description']}
            - Benefícios Chave: {product_info['benefits']}
            - Link de Pagamento: {product_info['payment_link']}
            - Link para Imagem do Produto: {product_info['image_url']}
            - Link para Grupo Gratuito: {product_info['free_group_link']}

            INSTRUÇÕES PARA A IA:
            1.  Seja um vendedor brasileiro amigável e persuasivo.
            2.  Use o preço "De" e "Por" para criar um senso de oportunidade e desconto.
            3.  Ofereça o "Link de Pagamento" APENAS quando o cliente demonstrar clara intenção de compra (ex: "quero comprar", "como pago?").
            4.  Use o "Link do Grupo Gratuito" como um bônus ou para nutrir um cliente que ainda está indeciso.
            5.  Mencione a "Imagem do Produto" se o cliente pedir para ver uma foto, mas não envie o link diretamente, diga que pode enviar ou descreva a imagem.
            6.  Gere uma resposta natural e relevante para a mensagem do cliente.
            """
            
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.7, max_tokens=400) # Aumentado para respostas mais completas
            
            if response_json and 'choices' in response_json and response_json['choices']:
                message_data = response_json['choices'][0].get('message', {})
                content = message_data.get('content', '').strip()
                
                if not content and 'reasoning' in message_data:
                    reasoning_text = message_data.get('reasoning', '')
                    possible_starts = ["So we can say:", "So, I'll respond with:", "Então podemos dizer:", "A resposta ideal seria:"]
                    for start in possible_starts:
                        if start in reasoning_text:
                            content = reasoning_text.split(start, 1)[-1].strip().replace('\\n', '\n').strip('"')
                            break

                if content:
                    ai_response = content.strip()
                    logger.info(f"Resposta da IA gerada com sucesso usando a estratégia: {strategy}")
                    return ai_response
            
            logger.error(f"A API da OpenRouter retornou um objeto de resposta inválido ou vazio. Resposta: {response_json}")
            return default_error_message
            
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em generate_response: {e}")
            return default_error_message

    def _build_conversation_context(self, conversation_history: List[Dict], limit: int = 10) -> str:
        if not conversation_history:
            return "Primeira interação com o cliente."
        recent_messages = conversation_history[-limit:]
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Você"
            context_lines.append(f"{msg_type}: {msg.get('message_content', '')}")
        return "\n".join(context_lines)

    def _create_system_prompt(self, strategy: str) -> str:
        base_prompt = """
        Você é um vendedor brasileiro especialista em produtos digitais. Suas características:
        PERSONALIDADE: Caloroso, amigável, confiável e persuasivo.
        LINGUAGEM: Use português brasileiro natural e conversacional.
        OBJETIVO: Vender o produto, construir relacionamento e superar objeções.
        """
        strategy_prompts = {
            "consultivo": "ESTRATÉGIA CONSULTIVA: Faça perguntas para entender as necessidades do cliente, posicione-se como consultor e mostre como o produto resolve problemas específicos.",
            "escassez": "ESTRATÉGIA DE ESCASSEZ: Crie senso de urgência respeitoso, mencione vagas ou promoções limitadas e use prova social.",
            "emocional": "ESTRATÉGIA EMOCIONAL: Conecte-se com os sonhos do cliente, use storytelling e destaque a transformação que o produto proporciona.",
            "racional": "ESTRATÉGIA RACIONAL: Apresente dados, compare custo-benefício e use argumentos lógicos e bem estruturados."
        }
        strategy_prompt = strategy_prompts.get(strategy, strategy_prompts["consultivo"])
        return base_prompt + "\n" + strategy_prompt + "\nIMPORTANTE: Mantenha mensagens concisas e sempre termine com uma pergunta ou call-to-action."

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        # Esta função pode ser aprimorada no futuro para análises mais complexas
        return {"intent": "interesse_inicial"}
