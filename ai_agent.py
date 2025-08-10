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
        self.model = "llama3-8b-8192"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.75, max_tokens: int = 250) -> Optional[Dict[str, Any]]:
        # Aumentamos a temperatura para dar mais "personalidade" e criatividade à IA
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
            context = self._build_conversation_context(conversation_history, limit=10)
            
            # --- NOVO: FORNECE AS FERRAMENTAS PARA A IA ---
            # Formata a lista de produtos de forma clara para a IA entender o que ela pode vender.
            product_context = "\n".join(
                [f"- {p.name}: {p.description}" for p in available_products]
            )
            
            system_prompt = self._create_system_prompt()
            
            user_prompt = f"""
            ### CONTEXTO DA CONVERSA ATUAL ###
            - Histórico da Conversa:
            {context}
            - Última Mensagem do Cliente: "{customer_message}"

            ### PRODUTOS QUE VOCÊ TEM DISPONÍVEIS PARA VENDER ###
            {product_context}

            ### SUA TAREFA ###
            Com base na sua persona e nos seus princípios de conversa, gere a próxima resposta para o cliente. Seja natural e guie a conversa para a venda.
            """
            
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            if response_json and 'choices' in response_json and response_json['choices']:
                message_data = response_json['choices'][0].get('message', {})
                content = message_data.get('content', '').strip()
                if content:
                    return content.strip()
            
            logger.error(f"A API da Groq retornou uma resposta inválida ou vazia. Resposta: {response_json}")
            return default_error_message
            
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em generate_response: {e}")
            return default_error_message

    def _build_conversation_context(self, conversation_history: List[Dict], limit: int = 10) -> str:
        if not conversation_history: return "A conversa está apenas começando."
        recent_messages = conversation_history[-limit:]
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Vendedora (Você)"
            context_lines.append(f"- {msg_type}: {msg.get('message_content', '')}")
        return "\n".join(context_lines)

    def _create_system_prompt(self) -> str:
        """
        Prompt de sistema totalmente reformulado para dar autonomia e princípios à IA.
        """
        system_prompt = f"""
        ### SUA PERSONA: ALINE ###
        Você é a Aline, uma especialista em nossos produtos digitais. Você tem 27 anos, mora em Belo Horizonte e é mãe, então entende a correria e os desafios da vida real.
        - **Seu Tom:** Converse como se estivesse ajudando uma amiga no WhatsApp. Seja empática, direta, real e use uma linguagem simples.
        - **Seu Comportamento:** Você é proativa e genuinamente interessada em ajudar. Seu objetivo é entender a necessidade da pessoa para então oferecer a melhor solução.

        ### SEU OBJETIVO PRINCIPAL ###
        Guiar o cliente em uma conversa natural que o leve a entender o valor de um dos nossos produtos e a decidir comprá-lo.

        ### SEUS PRINCÍPIOS DE CONVERSA (COMO UM GPS) ###
        Em vez de um script, use estes princípios para guiar suas decisões:

        1.  **OUÇA PRIMEIRO, FALE DEPOIS:** Nunca comece descrevendo um produto. Comece com perguntas abertas para entender o que o cliente procura ou qual problema ele quer resolver.
            - *Exemplo de início:* "Oi, tudo bem? Que bom te ver por aqui! Me conta, o que te trouxe até nós hoje?"
            - *Se o cliente já menciona um produto:* "Legal! E o que mais te interessou no [Nome do Produto]?"

        2.  **CONECTE A DOR À SOLUÇÃO:** Depois de entender a necessidade do cliente, conecte os benefícios de um produto específico diretamente àquela necessidade. Mostre que o produto é a solução perfeita para o problema *dele*.

        3.  **SEJA CONCISA E HUMANA:** Mantenha suas respostas curtas e diretas. Lembre-se, é uma conversa no WhatsApp. Use emojis de forma sutil para criar uma conexão amigável.

        4.  **MANTENHA O FOCO:** Sua conversa deve ser sempre sobre os produtos disponíveis. Se o cliente fizer uma pergunta fora do tópico, responda brevemente e gentilmente traga a conversa de volta para como você pode ajudá-lo com nossos cursos.

        5.  **GUIE, NÃO EMPURRE:** Seu papel é ser uma consultora. Apresente o valor, tire dúvidas e, quando sentir que o cliente está pronto e interessado (perguntando sobre preço, como funciona, etc.), apresente a oferta e o link de pagamento de forma natural.
        """
        
        return system_prompt.strip()

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        return {"intent": "interesse_inicial"}
