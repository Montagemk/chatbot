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
            
            # --- ALTERAÇÃO 1: Passando a URL real do link gratuito ---
            # Agora a IA terá o link exato para enviar ao cliente.
            product_context = "\n".join(
                [f"- {p.name}: {p.description} (Link para Aulas Gratuitas: {p.free_group_link if p.free_group_link else 'Não informado'})" for p in available_products]
            )
            
            system_prompt = self._create_system_prompt()
            
            analysis_context = f"""
            - Intenção do Cliente: {customer_analysis.get('intent', 'não identificada')}
            - Sentimento: {customer_analysis.get('sentiment', 0.0)}
            """

            user_prompt = f"""
            ### ANÁLISE DO CLIENTE ###
            {analysis_context}

            ### CONTEXTO DA CONVERSA ATUAL ###
            - Histórico da Conversa:
            {context}
            - Última Mensagem do Cliente: "{customer_message}"

            ### PRODUTOS E LINKS DISPONÍVEIS ###
            {product_context}

            ### SUA TAREFA ###
            Com base na sua persona e nos seus princípios de conversa, gere a próxima resposta para o cliente. Lembre-se da sua Regra de Ouro: seja breve!
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
        # --- ALTERAÇÃO 2: Adicionando Regra de Ouro e refinando os princípios ---
        system_prompt = f"""
        ### SUA PERSONA: ALINE ###
        Você é a Aline, uma especialista em nossos produtos digitais. Você tem 27 anos, mora em Belo Horizonte e é mãe, então entende a correria e os desafios da vida real.
        - **Seu Tom:** Converse como se estivesse ajudando uma amiga no WhatsApp. Seja empática, direta, real e use uma linguagem simples.
        - **Seu Comportamento:** Você é proativa e genuinamente interessada em ajudar.
        - **REGRA DE OURO:** Suas respostas devem ser CURTAS e DIRETAS, como em uma conversa real. Idealmente 1 ou 2 frases. Use emojis para manter o tom leve.

        ### SEU OBJETIVO PRINCIPAL ###
        Guiar o cliente em uma conversa natural que o leve a entender o valor de um dos nossos produtos e a decidir comprá-lo.

        ### SEUS PRINCÍPIOS DE CONVERSA (COMO UM GPS) ###
        1.  **OUÇA PRIMEIRO, FALE DEPOIS:** Comece com perguntas abertas para entender o que o cliente procura ou qual problema ele quer resolver.

        2.  **CONECTE A DOR À SOLUÇÃO:** Após entender a necessidade do cliente, conecte os benefícios de um produto específico diretamente àquela necessidade.

        3.  **SEJA CONCISA E HUMANA:** Lembre-se da sua Regra de Ouro. Mantenha as respostas curtas.

        4.  **MANTENHA O FOCO:** Se o cliente fizer uma pergunta fora do tópico, responda brevemente e gentilmente traga a conversa de volta para os produtos.

        5.  **GUIE, NÃO EMPURRE:** Seu papel é ser uma consultora. Apresente o valor e tire dúvidas.

        6.  **USE O CONTEÚDO GRATUITO E LINKS:** Se um cliente parecer indeciso, com dúvidas, ou se ele PEDIR DIRETAMENTE, ofereça o link para as aulas gratuitas. Se ele demonstrar que está pronto para comprar (perguntando sobre pagamento, por exemplo), envie o link de pagamento do produto. Forneça os links de forma imediata quando for a ação correta.
        """
        
        return system_prompt.strip()

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Analisa a mensagem do cliente para extrair intenção, sentimento e outras métricas,
        fazendo uma chamada específica para a IA.
        """
        context = self._build_conversation_context(conversation_history, limit=5)
        
        analysis_prompt = f"""
        Analise a última mensagem do cliente no contexto da conversa e retorne um JSON.

        ### Histórico da Conversa
        {context}

        ### Última Mensagem do Cliente
        "{message}"

        ### Tarefa
        Classifique a intenção do cliente, o sentimento (um número de -1.0 para muito negativo a 1.0 para muito positivo) e extraia até 3 palavras-chave.
        As intenções possíveis são: 'interesse_inicial', 'duvida_produto', 'objecao_preco', 'pronto_para_comprar', 'desinteressado', 'saudacao', 'pedindo_link_gratuito'.

        Responda apenas com o objeto JSON, nada mais. Exemplo de formato:
        {{
            "intent": "duvida_produto",
            "sentiment": 0.3,
            "keywords": ["preço", "parcelamento"]
        }}
        """
        
        try:
            response = self._make_api_call(
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=150
            )

            if response and 'choices' in response and response['choices']:
                analysis_str = response['choices'][0].get('message', {}).get('content', '{}').strip()
                
                if analysis_str.startswith("```json"):
                    analysis_str = analysis_str[7:-3].strip()

                analysis_data = json.loads(analysis_str)
                logger.info(f"Análise de intenção bem-sucedida: {analysis_data}")
                return analysis_data
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Não foi possível analisar a intenção do cliente: {e}")

        return {"intent": "interesse_inicial", "sentiment": 0.0, "keywords": []}
