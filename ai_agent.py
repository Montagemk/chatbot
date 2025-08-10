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
        Inicializa o Agente de IA com o modelo da Groq.
        """
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "deepseek-r1-distill-llama-70b"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 150) -> Optional[Dict[str, Any]]:
        # Esta função não precisa de alterações
        if not self.api_key:
            logger.error("A chave da API Groq não está configurada.")
            return None
        try:
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
            Siga RIGOROSAMENTE as regras da sua persona (Alin) e do seu fluxo de vendas. Gere APENAS a próxima mensagem para o cliente.
            """
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.75)

            if response_json and 'choices' in response_json and response_json['choices']:
                message_data = response_json['choices'][0].get('message', {})
                content = message_data.get('content', '').strip()

                if content:
                    # --- INÍCIO DA "GUARDA DE CONTENÇÃO" ---
                    # Esta é a trava de segurança.
                    # Se a IA enviar tags <think>, o código vai removê-las antes de retornar a resposta.
                    # Isso garante que o cliente NUNCA veja os pensamentos da IA.
                    if "<think>" in content and "</think>" in content:
                        # Pega apenas o texto que vem DEPOIS da tag de pensamento
                        content_parts = content.split("</think>")
                        if len(content_parts) > 1:
                            ai_response = content_parts[-1].strip()
                        else:
                            # Fallback caso algo estranho aconteça
                            ai_response = content.replace("<think>", "").replace("</think>", "").strip()
                    else:
                        ai_response = content
                    # --- FIM DA "GUARDA DE CONTENÇÃO" ---

                    if not ai_response:
                        logger.error(f"A IA gerou apenas pensamentos, sem uma resposta final. Conteúdo original: {content}")
                        return default_error_message

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
        """
        Prompt de sistema com instrução final ultra explícita.
        """
        strategy_text_map = { "consultivo": "educação", "escassez": "urgência", "emocional": "conexão", "racional": "lógica" }
        strategy_flavor = strategy_text_map.get(strategy, "educação")
        product_list_str = ", ".join(product_names)

        system_prompt = f"""
        ### SUA PERSONA: ALIN ###
        Você é a Alin, 27 anos, de BH, mãe. Converse como se estivesse no WhatsApp com uma amiga.
        - TOM DE VOZ: Empático, direto, real.
        - COMPORTAMENTO: Prática e focada em ajudar.

        ### SUAS REGRAS DE OURO ###
        1.  **UMA MENSAGEM DE CADA VEZ:** Envie sua resposta e **ESPERE** o cliente responder.
        2.  **SEJA CONCISA:** Respostas curtas, 1-2 frases
