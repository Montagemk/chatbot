import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configura o logger para este módulo.
logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        """
        Inicializa o Agente de IA, configurando chaves de API, endpoints e informações do produto.
        """
        # Configuração da API OpenRouter a partir de variáveis de ambiente
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # --- MODELO ATUALIZADO PARA 'openai/gpt-oss-20b' ---
        self.model = "openai/gpt-oss-20b"
        
        # Cabeçalhos HTTP necessários para a chamada da API OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("RENDER_EXTERNAL_URL", "https://comunidadeatp.store"),
            "X-Title": "WhatsApp AI Sales Agent"
        }
        
        # As informações do produto podem ser carregadas de variáveis de ambiente para maior flexibilidade
        self.product_info = {
            "name": os.environ.get("PRODUCT_NAME", "Curso Digital de Marketing"),
            "price": float(os.environ.get("PRODUCT_PRICE", "297.00")),
            "description": os.environ.get("PRODUCT_DESCRIPTION", "Aprenda as estratégias mais eficazes de marketing digital"),
            "benefits": [
                "Aumento de vendas em até 300%",
                "Suporte vitalício",
                "Certificado reconhecido",
                "Acesso a comunidade exclusiva"
            ]
        }
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 500) -> Optional[Dict[str, Any]]:
        """
        Realiza uma chamada robusta à API da OpenRouter com tratamento de erros e logging aprimorados.
        """
        if not self.api_key:
            logger.error("A chave da API OpenRouter não está configurada. Defina a variável de ambiente OPENROUTER_API_KEY.")
            return None
            
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 25  # Timeout do lado do servidor da API para a geração
            }
            
            # Timeout do lado do cliente para não prender o worker do Flask
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30  # Timeout de 30 segundos para a requisição
            )
            
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
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de requisição ao chamar a API OpenRouter: {e}")
            return None
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em _make_api_call: {e}")
            return None

    def analyze_customer_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Analisa a intenção do cliente a partir da sua mensagem e do histórico da conversa.
        """
        try:
            context = self._build_conversation_context(conversation_history)
            prompt = f"""
            Você é um especialista em análise de intenção de clientes brasileiros. 
            Analise a mensagem do cliente e determine:
            
            Contexto da conversa: {context}
            Mensagem atual: {message}
            
            Responda APENAS em formato JSON válido com:
            {{
                "intent": "interesse_inicial",
                "sentiment": 0.0,
                "urgency": 0.5,
                "keywords": ["palavra1", "palavra2"],
                "objections": [],
                "buying_signals": []
            }}
            
            Valores para intent: interesse_inicial, duvidas, objecoes, pronto_comprar, desinteressado
            Valores para sentiment: número de -1 a 1 (negativo a positivo)
            Valores para urgency: número de 0 a 1 (baixa a alta urgência)
            """
            
            response_json = self._make_api_call([
                {"role": "system", "content": "Você é um especialista em análise de clientes brasileiros. Responda APENAS com JSON válido."},
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            if response_json and 'choices' in response_json and response_json['choices']:
                message_content = response_json['choices'][0].get('message', {}).get('content')
                if message_content:
                    # Remove cercas de código markdown se a IA as incluir
                    if message_content.strip().startswith("```json"):
                        message_content = message_content.strip()[7:-3].strip()
                    return json.loads(message_content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar o JSON da análise de intenção: {e}. Resposta recebida: {message_content}")
        except Exception as e:
            logger.error(f"Erro ao analisar a intenção do cliente: {e}")
            
        return {
            "intent": "interesse_inicial", "sentiment": 0.0, "urgency": 0.5,
            "keywords": [], "objections": [], "buying_signals": []
        }
    
    def generate_response(self, customer_message: str, customer_analysis: Dict, 
                         conversation_history: List[Dict], strategy: str = "adaptive") -> str:
        """
        Gera uma resposta persuasiva e ciente do contexto para o cliente.
        """
        default_error_message = "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        
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
            - Sentimento: {customer_analysis.get('sentiment', 0)}
            - Urgência: {customer_analysis.get('urgency', 0.5)}
            - Objeções: {customer_analysis.get('objections', [])}
            - Sinais de compra: {customer_analysis.get('buying_signals', [])}
            
            CONTEXTO DA CONVERSA:
            {context}
            
            MENSAGEM ATUAL DO CLIENTE:
            {customer_message}
            
            INFORMAÇÕES DO PRODUTO:
            - Nome: {self.product_info['name']}
            - Preço: R$ {self.product_info['price']:.2f}
            - Descrição: {self.product_info['description']}
            - Benefícios: {', '.join(self.product_info['benefits'])}
            
            Gere uma resposta persuasiva, natural e culturalmente adequada para brasileiros.
            Use técnicas de copywriting e seja empático. Mantenha tom conversacional.
            """
            
            response_json = self._make_api_call([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.7, max_tokens=300)
            
            if response_json and 'choices' in response_json and response_json['choices']:
                message_data = response_json['choices'][0].get('message', {})
                content = message_data.get('content')
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
        """
        Constrói uma string representando o histórico recente da conversa.
        """
        if not conversation_history:
            return "Primeira interação com o cliente."
        
        recent_messages = conversation_history[-limit:]
        context_lines = []
        for msg in recent_messages:
            msg_type = "Cliente" if msg.get('message_type') == 'incoming' else "Você"
            context_lines.append(f"{msg_type}: {msg.get('message_content', '')}")
        
        return "\n".join(context_lines)
    
    def _create_system_prompt(self, strategy: str) -> str:
        """
        Cria um prompt de sistema detalhado com base na estratégia de vendas escolhida.
        """
        base_prompt = """
        Você é um vendedor brasileiro especialista em produtos digitais. Suas características:
        
        PERSONALIDADE:
        - Caloroso e amigável, típico brasileiro
        - Confiável e profissional
        - Empático e bom ouvinte
        - Persuasivo sem ser agressivo
        
        LINGUAGEM:
        - Use português brasileiro natural
        - Inclua expressões brasileiras quando apropriado
        - Tom conversacional e próximo
        - Evite linguagem muito formal
        
        OBJETIVO:
        - Vender o produto digital oferecido
        - Construir relacionamento e confiança
        - Superar objeções de forma inteligente
        - Criar urgência quando apropriado
        """
        
        strategy_prompts = {
            "consultivo": """
            ESTRATÉGIA CONSULTIVA:
            - Faça perguntas para entender as necessidades do cliente
            - Posicione-se como consultor, não apenas vendedor
            - Mostre como o produto resolve problemas específicos
            - Use histórias de sucesso de outros clientes brasileiros
            """,
            "escassez": """
            ESTRATÉGIA DE ESCASSEZ:
            - Mencione vagas limitadas ou promoção por tempo limitado
            - Crie senso de urgência respeitoso
            - Use prova social (outros brasileiros comprando)
            - Destaque benefícios únicos e exclusivos
            """,
            "emocional": """
            ESTRATÉGIA EMOCIONAL:
            - Conecte-se com sonhos e aspirações do cliente
            - Use storytelling envolvente
            - Destaque transformação de vida que o produto proporciona
            - Seja empático com desafios financeiros brasileiros
            """,
            "racional": """
            ESTRATÉGIA RACIONAL:
            - Apresente dados e resultados concretos
            - Compare custo-benefício de forma clara
            - Mencione garantias e suporte
            - Use argumentos lógicos e bem estruturados
            """
        }
        
        strategy_prompt = strategy_prompts.get(strategy, strategy_prompts["consultivo"])
        
        return base_prompt + "\n" + strategy_prompt + """
        
        IMPORTANTE:
        - Sempre responda em português brasileiro
        - Mantenha mensagens concisas (máximo 2-3 parágrafos)
        - Termine com uma pergunta ou call-to-action quando apropriado
        - Adapte-se ao nível de interesse demonstrado pelo cliente
        - Procure usar a escada de sim como modelo de vendas
        """

    def detect_purchase_intent(self, message: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """
        Detecta se o cliente está pronto para comprar com base na sua mensagem.
        """
        try:
            context = self._build_conversation_context(conversation_history)
            
            prompt = f"""
            Analise se o cliente está pronto para comprar com base na conversa.
            
            Contexto: {context}
            Mensagem atual: {message}
            
            Responda em JSON:
            {{
                "ready_to_buy": true/false,
                "confidence": 0.8,
                "purchase_signals": ["quero comprar", "qual o link"],
                "next_action": "send_payment_link|continue_conversation|address_objections"
            }}
            """
            
            response_json = self._make_api_call([
                {"role": "system", "content": "Você é um especialista em detectar intenção de compra. Responda APENAS com JSON válido."},
                {"role": "user", "content": prompt}
            ], temperature=0.2)
            
            if response_json and 'choices' in response_json and response_json['choices']:
                message_content = response_json['choices'][0].get('message', {}).get('content')
                if message_content:
                    if message_content.strip().startswith("```json"):
                        message_content = message_content.strip()[7:-3].strip()
                    return json.loads(message_content)
            
        except Exception as e:
            logger.error(f"Erro ao detectar intenção de compra: {e}")

        return {
            "ready_to_buy": False, "confidence": 0.0,
            "purchase_signals": [], "next_action": "continue_conversation"
        }
