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
        # (Esta função permanece a mesma da versão anterior, sem alterações)
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
                         product: Optional[Product] = None) -> str: # NOVO: Parâmetro product
        """
        Gera uma resposta persuasiva usando o contexto do produto fornecido.
        """
        default_error_message = "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        
        # --- LÓGICA DE CONTEXTO DO PRODUTO ATUALIZADA ---
        if product:
            # decodifica o JSON de benefícios, se necessário
            try:
                benefits_list = json.loads(product.key_benefits)
                benefits_str = ', '.join(benefits_list)
            except (json.JSONDecodeError, TypeError):
                benefits_str = str(product.key_benefits) # Fallback caso não seja JSON

            product_info = {
                "name": product.name,
                "price": product.price,
                "description": product.description,
                "benefits": benefits_str
            }
        else:
            # Fallback para informações padrão se nenhum produto for passado
            product_info = {
                "name": "nosso produto digital",
                "price": 0,
                "description": "uma solução completa para suas necessidades.",
                "benefits": "suporte, acesso a comunidade e muito mais."
            }

        try:
            from reinforcement_learning import ReinforcementLearner
            learner = ReinforcementLearner()
            if strategy == "adaptive":
                strategy = learner.get_best_strategy(customer_analysis)
            
            context = self._build_conversation_context(conversation_history, limit=5)
            system_prompt = self._create_system_prompt(strategy)
            
            user_prompt = f"""
            ANÁLISE DO CLIENTE:
            - Intenção: {customer_analysis.get('intent', 'desconhecida')}
            
            CONTEXTO DA CONVERSA:
            {context}
            
            MENSAGEM ATUAL DO CLIENTE:
            {customer_message}
            
            INFORMAÇÕES DO PRODUTO PARA VENDER:
            - Nome: {product_info['name']}
            - Preço: R$ {product_info['price']:.2f}
            - Descrição: {product_info['description']}
            - Benefícios: {product_info['benefits']}
            
            Gere uma resposta persuasiva, natural e culturalmente adequada para brasileiros.
            """
            
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.7, max_tokens=300)
            
            # --- LÓGICA DE EXTRAÇÃO DE RESPOSTA CORRIGIDA ---
            if response_json and 'choices' in response_json and response_json['choices']:
                message_data = response_json['choices'][0].get('message', {})
                content = message_data.get('content', '').strip()
                
                # Se o 'content' estiver vazio, tenta extrair do 'reasoning'
                if not content and 'reasoning' in message_data:
                    reasoning_text = message_data.get('reasoning', '')
                    # A resposta parece estar após o último "So we can say:" ou similar
                    possible_starts = ["So we can say:", "So, I'll respond with:", "Então podemos dizer:"]
                    for start in possible_starts:
                        if start in reasoning_text:
                            # Pega o texto após o marcador e limpa
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

    # O resto do arquivo (funções _build_conversation_context, _create_system_prompt, etc.) permanece o mesmo
    # ... (cole o restante do seu arquivo ai_agent.py aqui)

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
            "consultivo": "ESTRATÉGIA CONSULTIVA: Faça perguntas, posicione-se como consultor e mostre como o produto resolve problemas específicos.",
            "escassez": "ESTRATÉGIA DE ESCASSEZ: Crie senso de urgência respeitoso, mencione vagas ou promoções limitadas e use prova social.",
            "emocional": "ESTRATÉGIA EMOCIONAL: Conecte-se com os sonhos do cliente, use storytelling e destaque a transformação que o produto proporciona.",
            "racional": "ESTRATÉGIA RACIONAL: Apresente dados, compare custo-benefício e use argumentos lógicos e bem estruturados."
        }
        strategy_prompt = strategy_prompts.get(strategy, strategy_prompts["consultivo"])
        return base_prompt + "\n" + strategy_prompt + "\nIMPORTANTE: Mantenha mensagens concisas e sempre termine com uma pergunta ou call-to-action."

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        # Esta função pode ser simplificada por enquanto, já que o foco é a resposta
        return {"intent": "interesse_inicial"}
