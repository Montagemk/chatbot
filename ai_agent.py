import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from models import Product

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-8b-8192"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.75, max_tokens: int = 350) -> Optional[Dict[str, Any]]:
        # Aumentamos um pouco o max_tokens para comportar o JSON e a resposta
        if not self.api_key:
            logger.error("A chave da API Groq não está configurada.")
            return None
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"} # Força a resposta a ser um JSON
            }
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200 and response.text:
                return response.json()
            else:
                logger.error(f"Erro na API Groq: Status {response.status_code}. Resposta: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em _make_api_call: {e}")
            return None

    # --- FUNÇÃO PRINCIPAL OTIMIZADA PARA UMA ÚNICA CHAMADA ---
    def generate_response(self, customer_message: str, conversation_history: List[Dict], available_products: List[Product]) -> Dict[str, Any]:
        default_error_response = {
            "analysis": {"intent": "error", "sentiment": 0.0},
            "response": "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        }
        if not available_products:
            default_error_response["response"] = "Olá! No momento, estamos atualizando nosso catálogo."
            return default_error_response

        try:
            context = self._build_conversation_context(conversation_history, limit=10)
            product_context = "\n".join(
                [f"- {p.name}: {p.description} (Link para Aulas Gratuitas: {p.free_group_link if p.free_group_link else 'Não informado'})" for p in available_products]
            )
            
            system_prompt = self._create_system_prompt()
            
            user_prompt = f"""
            ### CONTEXTO DA CONVERSA ATUAL ###
            - Histórico da Conversa:
            {context}
            - Última Mensagem do Cliente: "{customer_message}"

            ### PRODUTOS E LINKS DISPONÍVEIS ###
            {product_context}

            ### SUA TAREFA ###
            Analise a última mensagem do cliente e gere uma resposta apropriada. Siga a sua persona e a Regra de Ouro.
            """
            
            response_json = self._make_api_call(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.75
            )

            if response_json and 'choices' in response_json and response_json['choices']:
                content_str = response_json['choices'][0].get('message', {}).get('content', '{}')
                structured_response = json.loads(content_str)
                
                # Valida se a resposta tem a estrutura esperada
                if 'analysis' in structured_response and 'response' in structured_response:
                    logger.info(f"Análise e resposta geradas com sucesso: {structured_response['analysis']}")
                    return structured_response
            
            logger.error(f"A API da Groq retornou um JSON inválido ou com estrutura incorreta. Resposta: {response_json}")
            return default_error_response
            
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em generate_response: {e}")
            return default_error_response

    def _build_conversation_context(self, conversation_history: List[Dict], limit: int = 10) -> str:
        if not conversation_history: return "A conversa está apenas começando."
        recent_messages = conversation_history[-limit:]
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Vendedora (Você)"
            context_lines.append(f"- {msg_type}: {msg.get('message_content', '')}")
        return "\n".join(context_lines)

    def _create_system_prompt(self) -> str:
        # --- PROMPT ATUALIZADO PARA GERAR UM JSON ESTRUTURADO ---
        system_prompt = f"""
        Você é a Aline, uma vendedora especialista em produtos digitais.
        
        ### SUA PERSONA ###
        - Tom: Amigável, empática, como uma amiga no WhatsApp.
        - Comportamento: Proativa, interessada em ajudar.
        - REGRA DE OURO: Suas respostas devem ser CURTAS e DIRETAS (1-2 frases).

        ### SUA TAREFA ###
        Sua tarefa é analisar a mensagem do cliente e formular a resposta da Aline.
        Você DEVE responder com um objeto JSON válido, contendo duas chaves: "analysis" e "response".

        1.  **analysis**: Um objeto contendo a sua análise da mensagem do cliente.
            - "intent": Classifique a intenção. Use um dos seguintes: 'interesse_inicial', 'duvida_produto', 'objecao_preco', 'pronto_para_comprar', 'desinteressado', 'saudacao', 'pedindo_link_gratuito'.
            - "sentiment": Um número de -1.0 a 1.0 representando o sentimento do cliente.

        2.  **response**: Uma string contendo a resposta que a Aline deve enviar para o cliente, seguindo a persona e a Regra de Ouro.

        ### EXEMPLO DE RESPOSTA JSON ###
        {{
          "analysis": {{
            "intent": "duvida_produto",
            "sentiment": 0.2
          }},
          "response": "Oi, tudo bem? O nosso curso custa X e você pode parcelar! Ajuda?"
        }}
        """
        return system_prompt.strip()

    # A função analyze_customer_intent não é mais necessária, pois a lógica foi unificada.
