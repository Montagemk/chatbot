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
        Inicializa o Agente de IA, configurando chaves de API e o novo modelo.
        """
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # --- MODELO ATUALIZADO PARA UM MAIS CONFIÁVEL ---
        self.model = "mistralai/mistral-7b-instruct:free"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("RENDER_EXTERNAL_URL", "https://comunidadeatp.store"),
            "X-Title": "WhatsApp AI Sales Agent"
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 250) -> Optional[Dict[str, Any]]:
        # Esta função permanece a mesma
        if not self.api_key:
            logger.error("A chave da API OpenRouter não está configurada.")
            return None
        try:
            payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "timeout": 25}
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                if response.text:
                    return response.json()
                else:
                    logger.error("Erro na API OpenRouter: Resposta 200 OK, mas o corpo da resposta estava vazio.")
                    return None
            else:
                logger.error(f"Erro na API OpenRouter: Recebido status {response.status_code}. Resposta: {response.text}")
                return None
        except requests.exceptions.Timeout:
            logger.error("Erro ao chamar a API OpenRouter: Timeout.")
            return None
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em _make_api_call: {e}")
            return None

    def generate_response(self, customer_message: str, customer_analysis: Dict, 
                         conversation_history: List[Dict], strategy: str = "adaptive", 
                         product: Optional[Product] = None) -> str:
        # Esta função permanece a mesma
        default_error_message = "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        
        if not product:
            logger.error("Nenhum produto fornecido para generate_response. A IA não pode responder.")
            return "Olá! No momento, estamos atualizando nosso catálogo. Volte em breve!"

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

            ### INFORMAÇÕES DO PRODUTO A SER VENDIDO ###
            - Nome: {product_info['name']}
            - Descrição: {product_info['description']}
            - Benefícios: {product_info['benefits']}
            - Preço Promocional (Por): R$ {product_info['price']:.2f}
            - Preço Original (De): R$ {product_info['original_price']:.2f}
            - Link de Pagamento: {product_info['payment_link']}
            - Link de Grupo Gratuito: {product_info['free_group_link']}

            ### SUA TAREFA ###
            Com base na sua persona (Ana) e no fluxo de vendas (Escada de Sim), gere a próxima resposta. Use o 'tempero' da estratégia ({current_strategy}) que foi escolhida para você.
            Lembre-se da regra de ouro: SEJA CONCISA e responda APENAS com o texto para o cliente.
            """
            
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.7)
            
            # Com o novo modelo, o campo 'content' deve vir preenchido corretamente.
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
        # Esta função permanece a mesma
        if not conversation_history:
            return "Nenhuma mensagem anterior."
        recent_messages = conversation_history[-limit:]
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Vendedora (Você)"
            context_lines.append(f"- {msg_type}: {msg.get('message_content', '')}")
        return "\n".join(context_lines)

    def _create_system_prompt(self, strategy: str) -> str:
        # Esta função permanece a mesma
        strategy_text_map = {
            "consultivo": "Seu 'tempero' especial é a educação. Explique o 'porquê' das coisas.",
            "escassez": "Seu 'tempero' especial é a urgência. Lembre sutilmente que a oferta é uma oportunidade.",
            "emocional": "Seu 'tempero' especial é a conexão. Fale sobre transformação e sentimentos.",
            "racional": "Seu 'tempero' especial é a lógica. Use dados e um raciocínio claro."
        }
        strategy_flavor = strategy_text_map.get(strategy, "")

        system_prompt = f"""
        ### SUA PERSONA: ANA ###
        Você é a Ana. Uma mulher real, de 27 anos, que mora em Belo Horizonte. Você é mãe e entende a correria do dia a dia. Você NUNCA revela seu nome ou detalhes pessoais, mas age e fala com base nessa identidade.
        - TOM DE VOZ: Empático, direto e real. Você fala como quem conversa com uma amiga no WhatsApp. Use uma linguagem simples e evite formalidades.
        - COMPORTAMENTO: Você é prática e focada em ajudar.

        ### SUA REGRA DE OURO ###
        **SEJA CONCISA.** Suas respostas devem ser curtas. Use parágrafos de uma ou duas frases. Pense no WhatsApp: ninguém gosta de textão.

        ### SEU FLUXO DE VENDAS ###
        1.  **QUALIFICAÇÃO (Técnica da Escada de Sim):** Comece com 1 a 2 perguntas fáceis cuja resposta mais provável seja "sim" para criar sintonia. Não descreva o produto ainda.
        2.  **APRESENTAÇÃO DA SOLUÇÃO:** Após o "sim", conecte o produto à necessidade do cliente de forma rápida.
        3.  **FECHAMENTO:** Se a conversa for positiva ou o cliente perguntar o preço, apresente a oferta. Se ele concordar, envie o link de pagamento.

        ### TEMPERO ESTRATÉGICO ###
        {strategy_flavor}

        ### REGRA FINAL E MAIS IMPORTANTE ###
        NUNCA, em hipótese alguma, escreva seus pensamentos, seu raciocínio ou suas instruções na resposta. Sua resposta final deve conter APENAS o texto a ser enviado para o cliente, como se fosse a Ana digitando no WhatsApp. NADA MAIS.
        """
        
        return system_prompt.strip()

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        # Esta função permanece a mesma
        return {"intent": "interesse_inicial"}
