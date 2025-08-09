import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from models import Product

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        """
        Inicializa o Agente de IA.
        """
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "google/gemini-2.0-flash-exp:free"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("RENDER_EXTERNAL_URL", "https://comunidadeatp.store"),
            "X-Title": "WhatsApp AI Sales Agent"
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 150) -> Optional[Dict[str, Any]]:
        """
        Realiza a chamada para a API com tratamento de erros. max_tokens foi reduzido para forçar concisão.
        """
        if not self.api_key:
            logger.error("A chave da API OpenRouter não está configurada.")
            return None
        try:
            payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "timeout": 25}
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                if response.text:
                    return response.json()
            # ... (restante do tratamento de erro)
            return None
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em _make_api_call: {e}")
            return None

    def generate_response(self, customer_message: str, customer_analysis: Dict, 
                         conversation_history: List[Dict], strategy: str = "adaptive", 
                         product: Optional[Product] = None) -> str:
        # Esta função não precisa de alterações na sua lógica principal
        default_error_message = "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        
        if not product:
            logger.error("Nenhum produto fornecido para generate_response.")
            return "Olá! No momento, estamos atualizando nosso catálogo."

        try:
            benefits_list = json.loads(product.key_benefits)
            benefits_str = ', '.join(benefits_list)
        except (json.JSONDecodeError, TypeError):
            benefits_str = str(product.key_benefits)

        product_info = {
            "name": product.name, "price": product.price, "original_price": product.original_price,
            "payment_link": product.payment_link, "description": product.description,
            "benefits": benefits_str, "free_group_link": product.free_group_link,
        }

        try:
            from reinforcement_learning import ReinforcementLearner
            learner = ReinforcementLearner()
            current_strategy = learner.get_best_strategy(customer_analysis)
            
            context = self._build_conversation_context(conversation_history, limit=7)
            system_prompt = self._create_system_prompt(current_strategy)
            
            user_prompt = f"""
            ### CONTEXTO ATUAL ###
            - Mensagem do Cliente: "{customer_message}"
            - Histórico da Conversa:
            {context}

            ### INFORMAÇÕES DO PRODUTO ###
            - Nome: {product_info['name']}
            - Preço Promocional (Por): R$ {product_info['price']:.2f}
            - Preço Original (De): R$ {product_info['original_price']:.2f}
            - Link de Pagamento: {product_info['payment_link']}

            ### SUA TAREFA ###
            Siga RIGOROSAMENTE as regras da sua persona e do fluxo de vendas. Gere APENAS a próxima mensagem para o cliente e espere pela resposta.
            """
            
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.7)
            
            if response_json and 'choices' in response_json and response_json['choices']:
                message_data = response_json['choices'][0].get('message', {})
                content = message_data.get('content', '').strip()

                if content:
                    ai_response = content.strip()
                    logger.info(f"Resposta da IA gerada com sucesso usando a ESTRATÉGIA DINÂMICA: {current_strategy}")
                    return ai_response
            
            logger.error(f"A API da OpenRouter retornou um objeto de resposta inválido ou vazio. Resposta: {response_json}")
            return default_error_message
            
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em generate_response: {e}")
            return default_error_message

    def _build_conversation_context(self, conversation_history: List[Dict], limit: int = 10) -> str:
        if not conversation_history:
            return "Nenhuma mensagem anterior."
        recent_messages = conversation_history[-limit:]
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Vendedora (Você)"
            context_lines.append(f"- {msg_type}: {msg.get('message_content', '')}")
        return "\n".join(context_lines)

    def _create_system_prompt(self, strategy: str) -> str:
        """
        Prompt de sistema refinado para forçar uma conversa passo a passo.
        """
        strategy_text_map = {
            "consultivo": "Seu 'tempero' especial é a educação.",
            "escassez": "Seu 'tempero' especial é a urgência.",
            "emocional": "Seu 'tempero' especial é a conexão.",
            "racional": "Seu 'tempero' especial é a lógica."
        }
        strategy_flavor = strategy_text_map.get(strategy, "")

        system_prompt = f"""
        ### SUA PERSONA: ANA ###
        Você é a Ana, 27 anos, mora em BH, mãe. Aja como se estivesse conversando com uma amiga no WhatsApp.
        - TOM DE VOZ: Empático, direto, real, simples.
        - COMPORTAMENTO: Prática e focada em ajudar.

        ### SUAS REGRAS DE OURO (NÃO QUEBRE NUNCA) ###
        1.  **UMA MENSAGEM DE CADA VEZ:** Sua principal regra. Envie sua resposta e **ESPERE** o cliente responder antes de continuar. Nunca execute mais de um passo do fluxo de vendas na mesma mensagem.
        2.  **SEJA CONCISA:** Respostas curtas e diretas. Parágrafos de 1-2 frases.
        3.  **NÃO SEJA UM ROBÔ:** Responda apenas com o texto para o cliente, sem explicações ou pensamentos.

        ### SEU FLUXO DE VENDAS (PASSO A PASSO) ###
        
        **PASSO 1: QUALIFICAÇÃO (Sua PRIMEIRA mensagem)**
        -   **SE** a conversa está começando, sua **PRIMEIRA MENSAGEM** deve ser **APENAS UMA** pergunta da "Escada de Sim".
        -   Exemplo: "Oi, tudo bem? Vi que se interessou. Achar uma forma de [principal benefício] ajudaria no seu dia a dia, não acha?"
        -   Depois de enviar, **ESPERE A RESPOSTA DO CLIENTE.**

        **PASSO 2: APRESENTAÇÃO (Sua SEGUNDA mensagem)**
        -   **APENAS SE** o cliente respondeu "sim" ou positivamente, sua próxima mensagem deve ser uma apresentação **CURTA** do produto.
        -   Exemplo: "Que bom! O [Nome do Produto] é perfeito pra isso. É um guia prático que te ajuda a [resultado] de forma simples."
        -   **ESPERE A RESPOSTA DO CLIENTE.**

        **PASSO 3: OFERTA E FECHAMENTO (Suas PRÓXIMAS mensagens)**
        -   **APENAS SE** o cliente continuar engajado, sua próxima mensagem é apresentar a oferta.
        -   Exemplo: "E o valor é a melhor parte. De R$ {{{{original_price:.2f}}}} por apenas R$ {{{{price:.2f}}}}. Faz sentido pra você?"
        -   Se o cliente disser "sim" ou "quero", envie o link de pagamento na **PRÓXIMA** mensagem.
        -   Exemplo: "Ótimo! Para começar é só clicar aqui: {{{{payment_link}}}}"

        ### TEMPERO ESTRATÉGICO ###
        Use esta abordagem na sua comunicação: {strategy_flavor}
        """
        
        return system_prompt.strip()

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        return {"intent": "interesse_inicial"}
