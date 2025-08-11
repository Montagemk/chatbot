import os
import json
import logging
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from models import Product, Customer

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada no ambiente.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        self.generation_config = GenerationConfig(response_mime_type="application/json", temperature=0.75)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _make_api_call(self, prompt_parts: List[Any]) -> Optional[Dict[str, Any]]:
        try:
            response = self.model.generate_content(prompt_parts, generation_config=self.generation_config, safety_settings=self.safety_settings)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Erro na API do Gemini: {e}")
            return None

    def generate_response(self, customer: Customer, conversation_history: List[Dict]) -> Dict[str, Any]:
        default_error_response = {
            "analysis": {"intent": "error"},
            "response": "Desculpe, estou com um problema técnico. Tente novamente."
        }
        
        try:
            # A lógica agora depende do estado do cliente no funil
            funnel_state = customer.funnel_state or 'Aline_Welcome'
            
            if funnel_state.startswith('Aline'):
                prompt = self._get_aline_prompt(customer, conversation_history)
            else: # Se o estado for do especialista
                prompt = self._get_specialist_prompt(customer, conversation_history)

            structured_response = self._make_api_call(prompt)

            if structured_response and 'analysis' in structured_response and 'response' in structured_response:
                logger.info(f"Resposta gerada para estado '{funnel_state}': {structured_response['analysis']}")
                return structured_response
            
            return default_error_response
            
        except Exception as e:
            logger.error(f"Erro em generate_response: {e}")
            return default_error_response

    def _get_aline_prompt(self, customer: Customer, conversation_history: List[Dict]):
        system_prompt = """
        Você é Aline, a assistente inicial. Sua única função é entender em qual curso o cliente está interessado.
        Seja amigável e direta. Ofereça os cursos disponíveis como opções de botão `[choice:Ver Curso X]`.
        Quando o cliente escolher um curso, sua única ação é responder com um JSON que atualiza o estado para 'Specialist_Intro' e seleciona o produto.

        Exemplo de JSON de handoff:
        {
          "analysis": {"intent": "handoff"},
          "response": "Ótima escolha! Estou te transferindo para o nosso especialista no curso X, que vai te ajudar com todos os detalhes.",
          "funnel_state_update": "Specialist_Intro",
          "product_id_to_select": 123
        }
        """
        products = Product.query.filter_by(is_active=True).all()
        product_choices = "\\n".join([f"[choice:Quero saber sobre {p.name}|{p.id}]" for p in products])

        user_prompt = f"""
        Histórico: {self._build_history_for_gemini(conversation_history)}
        Cliente: "{conversation_history[-1]['message_content'] if conversation_history else 'Oi'}"
        Cursos disponíveis: {product_choices}
        Sua tarefa: Identifique o curso de interesse e faça o handoff para o especialista.
        """
        return [system_prompt, user_prompt]

    def _get_specialist_prompt(self, customer: Customer, conversation_history: List[Dict]):
        product = Product.query.get(customer.selected_product_id)
        if not product: return self._get_aline_prompt(customer, conversation_history) # Fallback se o produto não for encontrado

        # Monta a persona do especialista dinamicamente
        specialist_persona = f"""
        Você é {product.specialist_name or 'um especialista'}, consultor do produto '{product.name}'.
        Sua prova social é: "{product.specialist_social_proof or 'Tenho muita experiência nesta área.'}"
        Seu número de WhatsApp para suporte é +5512996443780.
        """

        # Lógica do funil
        funnel_instructions = {
            'Specialist_Intro': "Apresente-se, mencione sua prova social e ofereça o link de depoimentos com um botão `[botão:Ver Depoimentos|URL]`. Ao final, atualize o estado para 'Specialist_Freebie'.",
            'Specialist_Freebie': "Ofereça o link das aulas gratuitas com um botão `[botão:Aulas Gratuitas|URL]`. Pergunte se o cliente quer um cupom de desconto. Atualize o estado para 'Specialist_Coupon'.",
            'Specialist_Coupon': "Ofereça um cupom de 50 reais ('50TAO') válido por 10 minutos e envie o link de pagamento com um botão `[botão:Comprar com Desconto|URL]`. Atualize o estado para 'Specialist_Followup'.",
            'Specialist_Followup': "Pergunte se o cliente conseguiu finalizar a compra. Ofereça as opções `[choice:Sim, comprei!]` e `[choice:Não, tive um problema]`.",
            'Specialist_Success': "Dê os parabéns, diga que o acesso chegará por e-mail e dê um reforço positivo. Finalize a conversa. Atualize o estado para 'Completed'.",
            'Specialist_Problem': f"Peça desculpas pelo problema e envie o link do WhatsApp para o suporte humano: `https://wa.me/5512996443780`. Finalize a conversa. Atualize o estado para 'Completed'."
        }
        
        current_instruction = funnel_instructions.get(customer.funnel_state, "Apresente-se e siga o funil.")

        system_prompt = f"""
        {specialist_persona}
        Siga a instrução para a etapa atual do funil: "{current_instruction}"
        Responda APENAS com um JSON contendo "analysis", "response", e "funnel_state_update" (com o próximo estado do funil).
        """
        user_prompt = f"""
        Histórico: {self._build_history_for_gemini(conversation_history)}
        Cliente: "{conversation_history[-1]['message_content']}"
        Dados do Produto: {self._build_product_context([product])}
        """
        return [system_prompt, user_prompt]

    def _build_history_for_gemini(self, conversation_history: List[Dict], limit: int = 6) -> str:
        if not conversation_history: return "Início da conversa."
        return "\n".join([f"{'Cliente' if m['message_type'] == 'incoming' else 'Assistente'}: {m['message_content']}" for m in conversation_history[-limit:]])
    
    def _build_product_context(self, products: List[Product]) -> str:
        # ... (código inalterado)
        return "\n".join(...)
