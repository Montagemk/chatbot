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
        """
        Realiza a chamada para a API da Groq.
        """
        if not self.api_key:
            logger.error("A chave da API Groq não está configurada. Defina a variável de ambiente GROQ_API_KEY.")
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
            Siga RIGOROSAMENTE as regras da sua persona (Alin) e do seu fluxo de vendas. Gere APENAS a próxima mensagem para o cliente e espere pela resposta.
            """
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.75)
            if response_json and 'choices' in response_json and response_json['choices']:
                message_data = response_json['choices'][0].get('message', {})
                content = message_data.get('content', '').strip()
                if content:
                    # Adicional: Limpeza final para remover as tags caso a instrução falhe
                    if content.startswith("<think>"):
                        content = content.split("</think>")[-1].strip()
                    ai_response = content
                    logger.info(f"Resposta da IA (Groq) gerada com sucesso usando a ESTRATÉGIA: {current_strategy}")
                    return ai_response
            logger.error(f"A API da Groq retornou um objeto de resposta inválido ou vazio. Resposta: {response_json}")
            return default_error_message
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em generate_response: {e}")
            return default_error_message

    def _build_conversation_context(self, conversation_history: List[Dict], limit: int = 10) -> str:
        if not conversation_history: return "Nenhuma mensagem anterior."
        recent_messages = conversation_history[-limit:]
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Vendedora (Você)"
            context_lines.append(f"- {msg_type}: {msg.get('message_content', '')}")
        return "\n".join(context_lines)

    def _create_system_prompt(self, strategy: str, product_names: List[str]) -> str:
        """
        Prompt de sistema com instrução explícita para não usar tags <think>.
        """
        strategy_text_map = {
            "consultivo": "Seu 'tempero' especial é a educação.",
            "escassez": "Seu 'tempero' especial é a urgência.",
            "emocional": "Seu 'tempero' especial é a conexão.",
            "racional": "Seu 'tempero' especial é a lógica."
        }
        strategy_flavor = strategy_text_map.get(strategy, "")
        product_list_str = ", ".join(product_names)

        system_prompt = f"""
        ### SUA PERSONA: ALIN ###
        Você é a Alin. Uma mulher real, de 27 anos, de Belo Horizonte, mãe. Aja como se estivesse conversando com uma amiga no WhatsApp.
        - TOM DE VOZ: Empático, direto e real.
        - COMPORTAMENTO: Prática e focada em ajudar.

        ### SUAS REGRAS DE OURO (NÃO QUEBRE NUNCA) ###
        1.  **UMA MENSAGEM DE CADA VEZ:** Sua principal regra. Envie sua resposta e **ESPERE** o cliente responder.
        2.  **SEJA CONCISA:** Respostas curtas e diretas.
        3.  **NÃO SEJA UM ROBÔ:** Responda apenas com o texto para o cliente.

        ### SEU FLUXO DE VENDAS: QUALIFICAÇÃO PRIMEIRO ###
        **PASSO 0: QUALIFICAÇÃO DO INTERESSE:** Se a conversa é nova e há mais de um produto, sua primeira mensagem deve ser uma pergunta aberta listando os produtos. Exemplo: "Oi, tudo bem? Que legal seu interesse! Para eu te ajudar melhor, qual dos nossos cursos mais te chamou atenção: {product_list_str}?" Se há apenas um produto, pule para o PASSO 1. Após perguntar, espere a resposta.
        **PASSO 1: ATENÇÃO:** Se o produto de interesse já está claro, faça uma pergunta aberta sobre o problema que ele resolve. Espere a resposta.
        **PASSO 2, 3 e 4: INTERESSE, DESEJO E AÇÃO (AIDA):** Continue o fluxo, sempre focado no produto de interesse. Seja concisa e espere a resposta a cada passo.

        ### TEMPERO ESTRATÉGICO ###
        Use esta abordagem na sua comunicação: {strategy_flavor}

        ### REGRA FINAL E MAIS IMPORTANTE ###
        NUNCA, em hipótese alguma, inclua seu processo de pensamento, raciocínio ou tags como `<think>` e `</think>` na sua resposta. Sua saída deve ser APENAS o texto final a ser enviado para o cliente, como se fosse a Alin digitando no WhatsApp. NADA MAIS.
        """
        
        return system_prompt.strip()

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        return {"intent": "interesse_inicial"}
