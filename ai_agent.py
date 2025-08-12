import os
import json
import logging
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

# Importamos os modelos para que o Agente possa consultar a base de dados diretamente
from models import Product, Customer 

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        # Configuração da API do Gemini (inalterada)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("A chave da API do Google não está configurada.")
            raise ValueError("GOOGLE_API_KEY não encontrada no ambiente.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        self.generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7 # Temperatura um pouco mais baixa para mais consistência
        )
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # --- ARQUITETURA DE MÁQUINA DE ESTADOS ---
        # Mapeia cada estado do funil a uma função específica que o trata.
        # Isto torna o código muito mais limpo e extensível.
        self.state_handlers = {
            'start': self._handle_start,
            'list_products': self._handle_list_products,
            'specialist_intro': self._handle_specialist_intro,
            'specialist_offer': self._handle_specialist_offer,
            'specialist_followup': self._handle_specialist_followup,
            'specialist_success': self._handle_specialist_success,
            'specialist_problem': self._handle_specialist_problem,
            'default': self._handle_default
        }

    def _make_api_call(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Função simplificada para fazer a chamada à API."""
        try:
            # A API do Gemini pode receber uma string simples diretamente.
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Erro na API do Gemini ou ao decodificar JSON: {e}")
            # Retorna um JSON de erro padrão
            return json.loads('{"text": "Desculpe, tive um problema para processar sua solicitação. Poderia tentar de novo?", "buttons": []}')

    def generate_response(self, customer: Customer, conversation_history: List[Dict], tactic: str) -> Dict[str, Any]:
        """
        Função principal que atua como um 'dispatcher'.
        Ela direciona a chamada para a função correta com base no estado do funil do cliente.
        """
        # Obtém o estado atual do funil do cliente.
        current_state = customer.funnel_state or 'start'
        
        # Encontra a função handler correspondente ao estado atual.
        # Se o estado não for encontrado, usa o handler padrão.
        handler = self.state_handlers.get(current_state, self._handle_default)
        
        logger.info(f"Cliente {customer.id} no estado '{current_state}'. A usar o handler: {handler.__name__}")
        
        # Executa a função handler para gerar a resposta estruturada.
        return handler(customer, conversation_history, tactic)

    # --- MÉTODOS HANDLER PARA CADA ESTADO DO FUNIL ---

    def _handle_start(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Gera a mensagem de boas-vindas inicial com as primeiras opções."""
        return {
            "text": "Olá! Sou o assistente virtual da Comunidade ATP. Como posso te ajudar hoje?",
            "buttons": [
                {"label": "Ver Cursos", "value": "Ver Cursos"},
                {"label": "Qual o Preço?", "value": "Qual o valor da comunidade?"},
                {"label": "WhatsApp", "value": "Quero falar no WhatsApp"}
            ],
            # Se o cliente escolher "Ver Cursos", o próximo estado será 'list_products'
            "funnel_state_update": "awaiting_choice" 
        }

    def _handle_list_products(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Busca os produtos ativos no banco de dados e os exibe como botões."""
        products = Product.query.filter_by(is_active=True).all()
        if not products:
            return {
                "text": "No momento, estamos atualizando nosso catálogo de cursos. Por favor, volte mais tarde!",
                "buttons": [],
                "funnel_state_update": "start"
            }

        product_buttons = [{"label": p.name, "value": f"Quero saber sobre o curso {p.id}"} for p in products]
        
        return {
            "text": "Ótima escolha! Temos os melhores especialistas do mercado. Qual destes cursos te interessa mais?",
            "buttons": product_buttons,
            "funnel_state_update": "awaiting_product_selection"
        }

    def _handle_specialist_intro(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Apresenta o especialista do produto selecionado."""
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._handle_default(customer, history, tactic, error="Produto não encontrado.")

        specialist_name = product.specialist_name or "um especialista"
        social_proof = product.specialist_social_proof or "posso te ajudar a alcançar seus objetivos."

        return {
            "text": f"Olá! Eu sou {specialist_name}, especialista no curso '{product.name}'. {social_proof}",
            "buttons": [
                {"label": "Acessar Aulas Gratuitas", "value": f"link:{product.free_group_link}"},
                {"label": "Ver Depoimentos", "value": f"link:{product.testimonials_link}"},
                {"label": "Gostei, quero a oferta!", "value": "Quero a oferta"}
            ],
            "funnel_state_update": "specialist_offer"
        }

    def _handle_specialist_offer(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Apresenta a oferta do produto com o cupom e link de pagamento."""
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._handle_default(customer, history, tactic, error="Produto não encontrado.")

        # Aqui, a IA poderia usar o 'tactic' para variar a forma da oferta.
        # Ex: se a tática for 'escassez', o texto poderia ser mais urgente.
        prompt = f"""
        Você é um vendedor especialista. Crie uma mensagem curta e poderosa oferecendo o produto '{product.name}' por R${product.price}.
        Mencione que o cupom '50TAO' dá 50% de desconto, mas é válido por apenas 10 minutos.
        Seja direto e persuasivo.
        Gere APENAS o texto da oferta em uma única string dentro de um JSON como este: {{"text": "sua_resposta_aqui"}}
        """
        response_json = self._make_api_call(prompt)
        offer_text = response_json.get("text", f"Aproveite a oferta especial para o curso {product.name}!")
        
        return {
            "text": offer_text,
            "buttons": [
                {"label": "✅ Comprar com Desconto", "value": f"link:{product.payment_link}"},
                {"label": "Tive um problema", "value": "Tive um problema na compra"},
                {"label": "Comprei!", "value": "Já comprei!"}
            ],
            "funnel_state_update": "specialist_followup"
        }

    def _handle_specialist_followup(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Lida com objeções usando a tática definida pelo ReinforcementLearner."""
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._handle_default(customer, history, tactic, error="Produto não encontrado.")

        # O 'tactic' vindo do routes.py (e definido pelo Learner) é crucial aqui.
        prompt = f"""
        Você é um especialista em vendas do produto '{product.name}'. O cliente demonstrou uma objeção ou problema.
        O histórico da conversa é: {history}.
        A sua tática para quebrar esta objeção é: '{tactic}'.
        Use esta tática para criar uma resposta empática e persuasiva, tentando resolver a dúvida do cliente.
        No final, incentive-o a tentar comprar novamente.
        Gere uma resposta em JSON como este: {{"text": "sua_resposta_aqui"}}
        """
        response_json = self._make_api_call(prompt)
        followup_text = response_json.get("text", "Entendo. Me diga, qual foi a sua principal dúvida para que eu possa te ajudar?")

        return {
            "text": followup_text,
            "buttons": [
                {"label": "Vou tentar comprar de novo", "value": f"link:{product.payment_link}"},
                {"label": "Ainda tenho dúvidas", "value": "Ainda estou com dúvidas"},
                {"label": "Falar com humano", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "specialist_followup"
        }
        
    def _handle_specialist_success(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Mensagem de parabéns após a compra."""
        return {
            "text": "Parabéns pela excelente decisão! Tenho a certeza de que você vai ter ótimos resultados. O acesso ao curso será enviado para o seu e-mail em instantes.",
            "buttons": [],
            "funnel_state_update": "completed"
        }

    def _handle_specialist_problem(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Lida com problemas na compra, direcionando para o suporte."""
        return {
            "text": "Sem problemas, estou aqui para ajudar! Para resolvermos isso o mais rápido possível, por favor, clique no botão abaixo para falar com nossa equipe de suporte no WhatsApp.",
            "buttons": [
                {"label": "Falar com Suporte", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "completed"
        }

    def _handle_default(self, customer: Customer, history: List[Dict], tactic: str, error: str = None) -> Dict[str, Any]:
        """Resposta padrão para situações não previstas ou erros."""
        error_message = error or "Não entendi muito bem. Como posso te ajudar?"
        return {
            "text": error_message,
            "buttons": [
                {"label": "Ver Cursos", "value": "Ver Cursos"},
                {"label": "Falar com Suporte", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "start" # Reinicia o fluxo em caso de erro
        }
