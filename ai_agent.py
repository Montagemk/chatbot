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
        Inicializa o Agente de IA com o modelo Llama 3 na Groq.
        """
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # --- 1. MODELO ATUALIZADO PARA LLAMA 3 (MAIS CONFIÁVEL) ---
        self.model = "llama3-8b-8192"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 350) -> Optional[Dict[str, Any]]:
        """
        Realiza a chamada para a API com mais espaço para a resposta.
        """
        if not self.api_key:
            logger.error("A chave da API Groq não está configurada.")
            return None
        try:
            # --- 2. AUMENTADO O ESPAÇO MÁXIMO DA RESPOSTA ---
            payload = { "model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens }
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200 and response.text:
                return response.json()
            else:
                logger.error(f"Erro na API Groq: Status {response.status_code}. Resposta: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em _make_api_call: {e}")
            return None

    def generate_response(self, customer_message: str, customer_analysis: Dict, 
                         conversation_history: List[Dict], available_products: List[Product]) -> str:
        # Esta função não precisa de alterações
        default_error_message = "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        if not available_products: return "Olá! No momento, estamos atualizando nosso catálogo."

        try:
            from reinforcement_learning import ReinforcementLearner
            learner = ReinforcementLearner()
            current_strategy = learner.get_best_strategy(customer_analysis)
            context = self._build_conversation_context(conversation_history, limit=7)
            product_names = [p.name for p in available_products]
            system_prompt = self._create_system_prompt(current_strategy, product_names)
            user_prompt = f"""
            ### CONTEXTO ATUAL ###
            - Mensagem do Cliente: "{customer_message}"
            - Histórico da Conversa:
            {context}

            ### SUA TAREFA ###
            Siga RIGOROSAMENTE as regras da sua persona (Aline) e do seu fluxo de vendas. Gere APENAS a próxima mensagem para o cliente.
            """
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.75)

            if response_json and 'choices' in response_json and response_json['choices']:
                message_data = response_json['choices'][0].get('message', {})
                content = message_data.get('content', '').strip()

                if content:
                    ai_response = content
                    logger.info(f"Resposta da IA (Groq) gerada com sucesso. Estratégia: {current_strategy}")
                    return ai_response
            
            logger.error(f"A API da Groq retornou uma resposta inválida ou vazia. Resposta: {response_json}")
            return default_error_message
            
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em generate_response: {e}")
            return default_error_message

    def _build_conversation_context(self, conversation_history: List[Dict], limit: int = 10) -> str:
        # Esta função não precisa de alterações
        if not conversation_history: return "Nenhuma mensagem anterior."
        recent_messages = conversation_history[-limit:]
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Vendedora (Você)"
            context_lines.append(f"- {msg_type}: {msg.get('message_content', '')}")
        return "\n".join(context_lines)

    def _create_system_prompt(self, strategy: str, product_names: List[str]) -> str:
        # Esta função não precisa de alterações
        strategy_text_map = { "consultivo": "educação", "escassez": "urgência", "emocional": "conexão", "racional": "lógica" }
        strategy_flavor = strategy_text_map.get(strategy, "educação")
        product_list_str = ", ".join(product_names)

        system_prompt = f"""
        ### SUA PERSONA: ALINE ###
        Você é a Aline. Uma mulher real, de 27 anos, que mora em Belo Horizonte. Você é mãe e entende a correria do dia a dia. Você NUNCA revela seu nome ou detalhes pessoais, mas age e fala com base nessa identidade.
        - TOM DE VOZ: Empático, direto e real. Você fala como quem conversa com uma amiga no WhatsApp.
        - COMPORTAMENTO: Prática e focada em ajudar.

        ### SUAS REGRAS DE OURO (NÃO QUEBRE NUNCA) ###
        1.  **UMA MENSAGEM DE CADA VEZ:** Sua principal regra. Envie sua resposta e **ESPERE** o cliente responder.
        2.  **SEJA CONCISA:** Respostas curtas e diretas.
        3.  **NÃO SEJA UM ROBÔ:** Sua resposta final deve ser APENAS o texto para o cliente. É PROIBIDO incluir seu raciocínio ou pensamentos.

        ### SEU FLUXO DE VENDAS PARA PRODUTOS DIGITAIS ###

        **PASSO 0: QUALIFICAÇÃO**
        Se a conversa for nova e houver vários produtos, pergunte qual deles o cliente quer conhecer.
        Ação: Espere a resposta do cliente antes de prosseguir.

        **PASSO 1: CONEXÃO E DOR**
        Faça uma pergunta curta e aberta que explore o problema ou desafio que o produto resolve.
        Ação: Espere o cliente descrever sua necessidade.

        **PASSO 2: SOLUÇÃO**
        Valide a dor do cliente e apresente seu produto como a solução ideal para aquele problema específico.
        Ação: Mostre que você entendeu o cliente e tem a resposta que ele procura.

        **PASSO 3: OFERTA E VALOR**
        Apresente o preço como uma oportunidade única, usando a ancoragem de preço "De/Por".
        Ação: Finalize com uma pergunta de validação como "Faz sentido pra você?".

        **PASSO 4: AÇÃO**
        Se o cliente der um sinal de compra claro, envie o link de pagamento de forma direta e prestativa.
        Ação: Facilite o próximo passo para o cliente fechar a compra.

        ### TEMPERO ESTRATÉGICO ###
        Use esta abordagem na sua comunicação: {strategy_flavor}
        """
        
        return system_prompt.strip()
        
        return system_prompt.strip()

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        return {"intent": "interesse_inicial"}
