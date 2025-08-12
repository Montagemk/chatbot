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
            temperature=0.7
        )
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # --- ARQUITETURA DE MÁQUINA DE ESTADOS ATUALIZADA ---
        # Adicionámos os novos estados de processamento ('awaiting_...')
        self.state_handlers = {
            'start': self._handle_start,
            'awaiting_choice': self._handle_awaiting_choice,
            'list_products': self._handle_list_products,
            'awaiting_product_selection': self._handle_awaiting_product_selection,
            'get_price': self._handle_get_price,
            'whatsapp_redirect': self._handle_whatsapp_redirect,
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
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Erro na API do Gemini ou ao decodificar JSON: {e}")
            return json.loads('{"text": "Desculpe, tive um problema para processar sua solicitação. Poderia tentar de novo?", "buttons": []}')

    def generate_response(self, customer: Customer, conversation_history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Função principal que atua como um 'dispatcher'."""
        current_state = customer.funnel_state or 'start'
        handler = self.state_handlers.get(current_state, self._handle_default)
        
        logger.info(f"Cliente {customer.id} no estado '{current_state}'. A usar o handler: {handler.__name__}")
        
        return handler(customer, conversation_history, tactic)

    # --- NOVOS HANDLERS DE PROCESSAMENTO ('Controladores de Tráfego') ---

    def _handle_awaiting_choice(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Controlador que direciona o fluxo após a escolha inicial do utilizador."""
        last_message = history[-1]['message_content'].lower() if history else ""

        if "ver cursos" in last_message:
            return self._handle_list_products(customer, history, tactic)
        elif "preço" in last_message or "valor" in last_message:
            return self._handle_get_price(customer, history, tactic)
        elif "whatsapp" in last_message:
            return self._handle_whatsapp_redirect(customer, history, tactic)
        else:
            return self._handle_default(customer, history, tactic)

    def _handle_awaiting_product_selection(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Controlador que processa a escolha do produto."""
        last_message = history[-1]['message_content'] if history else ""
        
        try:
            # Extrai o ID do produto da mensagem (ex: "Quero saber sobre o curso 5")
            product_id = int(last_message.split(' ')[-1])
            
            # Valida se o produto existe
            product = Product.query.get(product_id)
            if product:
                # Se o produto é válido, não devolvemos uma mensagem, mas sim instruções
                # para o routes.py atualizar a base de dados e chamar o próximo handler.
                return {
                    "text": None, # Sem texto, pois é uma ação interna
                    "buttons": [],
                    "product_id_to_select": product_id, # Instrução para routes.py
                    "funnel_state_update": "specialist_intro" # Próximo estado
                }
        except (ValueError, IndexError):
            logger.warning(f"Não foi possível extrair o ID do produto da mensagem: '{last_message}'")
        
        # Se algo falhar, volta a listar os produtos
        return self._handle_list_products(customer, history, tactic)


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
            "funnel_state_update": "awaiting_choice" 
        }

    def _handle_list_products(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Busca os produtos ativos no banco de dados e os exibe como botões."""
        products = Product.query.filter_by(is_active=True).all()
        if not products:
            return self._handle_default(customer, history, tactic, error="No momento, estamos atualizando nosso catálogo.")

        product_buttons = [{"label": p.name, "value": f"Quero saber sobre o curso {p.id}"} for p in products]
        
        return {
            "text": "Ótima escolha! Temos os melhores especialistas do mercado. Qual destes cursos te interessa mais?",
            "buttons": product_buttons,
            "funnel_state_update": "awaiting_product_selection"
        }
        
    def _handle_get_price(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Informa o preço da comunidade e oferece os próximos passos."""
        # Nota: Idealmente, o preço poderia vir de uma configuração geral, mas por agora está fixo.
        return {
            "text": "O acesso à Comunidade ATP com mais de 800 cursos é vitalício, por um pagamento único de R$97. O que você gostaria de fazer?",
            "buttons": [
                {"label": "Ver a Lista de Cursos", "value": "Ver Cursos"},
                {"label": "Falar com Suporte", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "awaiting_choice"
        }

    def _handle_whatsapp_redirect(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Confirma a intenção de ir para o WhatsApp e fornece o botão."""
        return {
            "text": "Claro! Para falar com um dos nossos consultores humanos, basta clicar no botão abaixo.",
            "buttons": [
                {"label": "Abrir WhatsApp", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "completed"
        }

    def _handle_specialist_intro(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """Apresenta o especialista do produto selecionado."""
        # Se esta função for chamada diretamente pelo routes.py após uma atualização,
        # o customer.selected_product_id estará atualizado.
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._handle_default(customer, history, tactic, error="Produto não encontrado.")

        specialist_name = product.specialist_name or "um especialista"
        social_proof = product.specialist_social_proof or "posso te ajudar a alcançar seus objetivos."

        return {
            "text": f"Perfeito! Eu sou {specialist_name}, especialista no curso '{product.name}'. {social_proof}\n\nO que você gostaria de ver primeiro?",
            "buttons": [
                {"label": "Acessar Aulas Gratuitas", "value": f"link:{product.free_group_link}"},
                {"label": "Ver Depoimentos", "value": f"link:{product.testimonials_link}"},
                {"label": "Gostei, quero a oferta!", "value": "Quero a oferta"}
            ],
            # O próximo estado lógico será processar a escolha sobre a oferta/aulas
            "funnel_state_update": "awaiting_offer_choice" 
        }

    def _handle_specialist_offer(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        # (Este handler e os seguintes permaneceriam muito semelhantes, pois já são estados de "Apresentação")
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._handle_default(customer, history, tactic, error="Produto não encontrado.")
        prompt = f"""
        Você é um vendedor especialista. Crie uma mensagem curta e poderosa oferecendo o produto '{product.name}' por R${product.price}.
        Mencione que o cupom '50TAO' dá 50% de desconto, mas é válido por apenas 10 minutos.
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
            "funnel_state_update": "awaiting_purchase_outcome"
        }

    # ... (Os handlers _handle_specialist_followup, _handle_specialist_success, etc. continuariam aqui) ...
    # Por agora, vamos simplificar e adicionar os que faltam
    def _handle_specialist_followup(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        return {"text": "Follow-up", "buttons": [], "funnel_state_update": "completed"}
        
    def _handle_specialist_success(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        return {"text": "Sucesso!", "buttons": [], "funnel_state_update": "completed"}

    def _handle_specialist_problem(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        return {"text": "Problema", "buttons": [], "funnel_state_update": "completed"}


    def _handle_default(self, customer: Customer, history: List[Dict], tactic: str, error: str = None) -> Dict[str, Any]:
        """Resposta padrão para situações não previstas ou erros."""
        error_message = error or "Não entendi muito bem. Como posso te ajudar?"
        return {
            "text": error_message,
            "buttons": [
                {"label": "Ver Cursos", "value": "Ver Cursos"},
                {"label": "Falar com Suporte", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "start" 
        }
