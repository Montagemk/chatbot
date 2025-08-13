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
        # Configuração da API do Gemini
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

        # --- ARQUITETURA DE MÁQUINA DE ESTADOS COMPLETA ---
        self.state_handlers = {
            'start': self._handle_start,
            'awaiting_choice': self._handle_awaiting_choice,
            'list_products': self._handle_list_products,
            'awaiting_product_selection': self._handle_awaiting_product_selection,
            'get_price': self._handle_get_price,
            'whatsapp_redirect': self._handle_whatsapp_redirect,
            'specialist_intro': self._handle_specialist_intro,
            'awaiting_offer_choice': self._handle_awaiting_offer_choice,
            'specialist_offer': self._handle_specialist_offer,
            'awaiting_purchase_outcome': self._handle_awaiting_purchase_outcome,
            
            # Lógica de Triagem Inteligente de Problemas
            'specialist_problem': self._handle_specialist_problem,
            'awaiting_problem_category': self._handle_awaiting_problem_category,
            'awaiting_specific_description': self._handle_awaiting_specific_description,

            'specialist_followup': self._handle_specialist_followup,
            'specialist_success': self._handle_specialist_success,
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

    # --- HANDLERS DE PROCESSAMENTO ('Controladores de Tráfego') ---

    def _handle_awaiting_choice(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
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
        last_message = history[-1]['message_content'] if history else ""
        try:
            # Extrai o ID do produto da mensagem "Quero saber sobre o curso {p.id}"
            product_id = int(last_message.split(' ')[-1])
            product = Product.query.get(product_id)
            if product:
                # Retorna instruções para o sistema principal atualizar o cliente e avançar o funil
                return { "text": None, "buttons": [], "product_id_to_select": product_id, "funnel_state_update": "specialist_intro" }
        except (ValueError, IndexError):
            logger.warning(f"Não foi possível extrair o ID do produto da mensagem: '{last_message}'")
        # Se falhar, lista os produtos novamente
        return self._handle_list_products(customer, history, tactic)

    def _handle_awaiting_offer_choice(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        last_message = history[-1]['message_content'].lower() if history else ""
        if "oferta" in last_message:
            return self._handle_specialist_offer(customer, history, tactic)
        else:
            return {
                "text": "Entendido. Quando estiver pronto para ver a oferta especial que preparei para ti, é só avisar!",
                "buttons": [{"label": "Estou pronto, quero a oferta!", "value": "Quero a oferta"}],
                "funnel_state_update": "awaiting_offer_choice"
            }

    def _handle_awaiting_purchase_outcome(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        last_message = history[-1]['message_content'].lower() if history else ""
        if "comprei" in last_message:
            return self._handle_specialist_success(customer, history, tactic)
        elif "problema" in last_message:
            return self._handle_specialist_problem(customer, history, tactic)
        else:
            return self._handle_specialist_followup(customer, history, tactic)
            
    def _handle_awaiting_problem_category(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """
        CÉREBRO DA TRIAGEM: Oferece soluções diferentes para cada categoria de problema.
        """
        last_message = history[-1]['message_content'].lower() if history else ""
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._handle_default(customer, history, tactic, error="Produto não encontrado para triagem.")

        # Lógica para Problemas de Pagamento
        if "pagamento" in last_message:
            return {
                "text": "Entendi, problemas com pagamento acontecem. Muitas vezes, tentar com outro cartão ou verificar os dados já resolve. Eu posso também gerar um novo link de pagamento para você, isso costuma funcionar!",
                "buttons": [
                    {"label": "✅ Sim, gerar novo link", "value": f"link:{product.payment_link}"},
                    {"label": "Falar com Suporte", "value": "Quero falar no WhatsApp"}
                ],
                "funnel_state_update": "awaiting_purchase_outcome"
            }
        
        # Lógica para Problemas de Link ou Cupom
        elif "link" in last_message or "cupom" in last_message:
            return {
                "text": "Claro! Às vezes o cupom precisa ser reaplicado ou o link expira. Acabei de gerar um novo link de compra com o desconto para você. Clique abaixo para tentar de novo.",
                "buttons": [
                    {"label": "Tentar comprar com novo link", "value": f"link:{product.payment_link}"}
                ],
                "funnel_state_update": "awaiting_purchase_outcome"
            }

        # Lógica para Dúvidas ou Outros Problemas (aqui pedimos para escrever)
        else: # "outra duvida"
            return {
                "text": "Entendido. Por favor, agora sim, escreva em detalhes a sua dúvida ou o problema que está a enfrentar para que eu possa te ajudar da melhor forma.",
                "buttons": [],
                "funnel_state_update": "awaiting_specific_description"
            }
            
    def _handle_awaiting_specific_description(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """
        ESCUTA ATIVA: Pega a descrição do problema e a envia para a IA Generativa.
        """
        return self._handle_specialist_followup(customer, history, tactic)

    # --- MÉTODOS HANDLER DE APRESENTAÇÃO ---

    def _handle_start(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        return {
            "text": "Olá! Sou da Comunidade ATP. Como posso te ajudar hoje?",
            "buttons": [
                {"label": "Ver Cursos", "value": "Ver Cursos"},
                {"label": "Qual o Preço?", "value": "Qual o valor da comunidade?"},
                {"label": "WhatsApp", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "awaiting_choice" 
        }

    def _handle_list_products(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
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
        return {
            "text": "O acesso à Comunidade ATP com mais de 800 cursos é vitalício, por um pagamento único. O que você gostaria de fazer?",
            "buttons": [
                {"label": "Ver a Lista de Cursos", "value": "Ver Cursos"},
                {"label": "Falar com Suporte", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "awaiting_choice"
        }

    def _handle_whatsapp_redirect(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        return {
            "text": "Claro! Para falar com um dos nossos consultores, basta clicar no botão abaixo.",
            "buttons": [
                {"label": "Abrir WhatsApp", "value": "Quero falar no WhatsApp"}
            ],
            "funnel_state_update": "completed"
        }

    def _handle_specialist_intro(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._handle_default(customer, history, tactic, error="Produto não encontrado.")
        specialist_name = product.specialist_name or "um especialista"
        social_proof = product.specialist_social_proof or "posso te ajudar a alcançar seus objetivos."
        return {
            "text": f"Perfeito! Eu sou {specialist_name}, especialista no curso '{product.name}'. {social_proof}\n\n O que você gostaria de ver primeiro?",
            "buttons": [
                {"label": "Acessar Aulas Gratuitas", "value": f"link:{product.free_group_link}"},
                {"label": "Ver Depoimentos", "value": f"link:{product.testimonials_link}"},
                {"label": "Gostei, quero a oferta!", "value": "Quero a oferta"}
            ],
            "funnel_state_update": "awaiting_offer_choice" 
        }

    def _handle_specialist_offer(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._handle_default(customer, history, tactic, error="Produto não encontrado.")
        prompt = f"""
        Você é um vendedor especialista. Crie uma mensagem curta e poderosa oferecendo o produto '{product.name}' por R${product.price}.
        Mencione que o cupom '50TAO' dá 50 Reais de desconto, mas é válido por apenas 10 minutos.
        Gere APENAS o texto da oferta em uma única string dentro de um JSON como este: {{"text": "sua_resposta_aqui"}}
        """
        response_json = self._make_api_call(prompt)
        offer_text = response_json.get("text", f" Aproveite a oferta especial para o curso {product.name}!")
        return {
            "text": offer_text,
            "buttons": [
                {"label": "✅ Comprar com Desconto", "value": f"link:{product.payment_link}"},
                {"label": "Tive um problema", "value": "Tive um problema na compra"},
                {"label": "Comprei!", "value": "Já comprei!"}
            ],
            "funnel_state_update": "awaiting_purchase_outcome"
        }

   def _handle_specialist_followup(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
    """
    PRIMEIRA TENTATIVA DE FOLLOW-UP: Lida com a primeira objeção e prepara para a tentativa final.
    """
    product = Product.query.get(customer.selected_product_id)
    if not product:
        return self._handle_default(customer, history, tactic, error="Produto não encontrado.")
    
    prompt = f"""
    Você é um especialista em vendas do produto '{product.name}'. O cliente tem uma dúvida ou objeção.
    O histórico da conversa é: {history}.
    A sua tática para quebrar esta objeção é: '{tactic}'.
    Use esta tática para criar uma resposta empática e persuasiva.
    Gere uma resposta em JSON como este: {{"text": "sua_resposta_aqui"}}
    """
    response_json = self._make_api_call(prompt)
    followup_text = response_json.get("text", "Entendo a sua dúvida. Deixe-me explicar melhor...")
    
    return {
        "text": followup_text,
        "buttons": [
            {"label": "✅ Entendi, quero comprar agora", "value": f"link:{product.payment_link}"},
            {"label": "Ainda tenho uma dúvida", "value": "Ainda tenho uma dúvida"}
        ],
        # MUDANÇA CRÍTICA: Movemos para um novo estado de "última tentativa"
        "funnel_state_update": "awaiting_final_objection" 
    }
        
    def _handle_specialist_success(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        return {
            "text": "Parabéns pela sua decisão e seja muito bem-vindo(a) à comunidade! Tenho a certeza de que vai aproveitar ao máximo. O seu acesso será enviado para o seu e-mail em instantes.",
            "buttons": [],
            "funnel_state_update": "completed"
        }

    def _handle_specialist_problem(self, customer: Customer, history: List[Dict], tactic: str) -> Dict[str, Any]:
        """
        INÍCIO DA TRIAGEM INTELIGENTE: Pergunta a categoria do problema ao cliente.
        """
        return {
            "text": "Sem problemas, estou aqui para ajudar! Para eu entender melhor, o seu problema está relacionado a quê?",
            "buttons": [
                {"label": "Problema com Pagamento", "value": "problema pagamento"},
                {"label": "Link ou Cupom não funciona", "value": "problema link cupom"},
                {"label": "Tenho uma outra dúvida", "value": "problema outra duvida"},
            ],
            "funnel_state_update": "awaiting_problem_category"
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
            "funnel_state_update": "start" 
        }
