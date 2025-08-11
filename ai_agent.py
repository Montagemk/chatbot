import os
import json
import logging
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

# A IA agora precisa conhecer os modelos Customer e Product
from models import Product, Customer

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("A chave da API do Google não está configurada.")
            raise ValueError("GOOGLE_API_KEY não encontrada no ambiente.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        self.generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.75
        )
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _make_api_call(self, prompt_parts: List[Any]) -> Optional[Dict[str, Any]]:
        try:
            response = self.model.generate_content(
                prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Erro na API do Gemini ou ao decodificar JSON: {e}")
            return None

    def generate_response(self, customer: Customer, conversation_history: List[Dict]) -> Dict[str, Any]:
        default_error_response = {
            "analysis": {"intent": "error", "sentiment": 0.0},
            "response": "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        }
        
        try:
            funnel_state = customer.funnel_state or 'Aline_Welcome'
            last_message = conversation_history[-1]['message_content'] if conversation_history else ''

            # Roteador: Decide qual persona (Aline ou Especialista) deve agir
            if funnel_state.startswith('Aline'):
                prompt_parts = self._get_aline_prompt(last_message, conversation_history)
            else:
                prompt_parts = self._get_specialist_prompt(customer, last_message, conversation_history)

            structured_response = self._make_api_call(prompt_parts)

            if structured_response and 'analysis' in structured_response and 'response' in structured_response:
                logger.info(f"Resposta gerada para estado '{funnel_state}': {structured_response.get('analysis')}")
                return structured_response
            
            logger.error(f"A API do Gemini retornou um JSON inválido ou com estrutura incorreta. Resposta: {structured_response}")
            return default_error_response
            
        except Exception as e:
            logger.error(f"Erro em generate_response: {e}", exc_info=True)
            return default_error_response

    def _get_aline_prompt(self, customer_message: str, conversation_history: List[Dict]):
        system_prompt = """
        Você é Aline, a assistente inicial. Sua ÚNICA função é descobrir qual curso interessa ao cliente e transferi-lo para um especialista.
        1.  Apresente-se e ofereça os cursos disponíveis como botões de escolha no formato `[choice:Quero saber sobre NOME_DO_CURSO|ID_DO_PRODUTO]`.
        2.  Quando o cliente escolher um curso (seja por texto ou clique), sua ÚNICA ação é responder com um JSON para fazer a transferência. A sua resposta de texto deve ser apenas a mensagem de transferência.

        Exemplo de JSON de transferência:
        {
          "analysis": {"intent": "handoff_specialist", "sentiment": 0.8},
          "response": "Ótima escolha! Estou te transferindo para o nosso especialista no curso de Copywriting, que vai tirar todas as suas dúvidas.",
          "funnel_state_update": "Specialist_Intro",
          "product_id_to_select": 123
        }
        """
        products = Product.query.filter_by(is_active=True).all()
        # O formato do botão agora inclui o ID do produto, que será usado pelo routes.py
        product_choices = "\\n".join([f"[choice:Quero saber sobre {p.name}|{p.id}]" for p in products])

        user_prompt = f"""
        Histórico da conversa: {self._build_history_for_gemini(conversation_history)}
        Última mensagem do Cliente: "{customer_message}"
        Sua tarefa: Ofereça os cursos abaixo como opções. Se o cliente já escolheu um, faça a transferência conforme o exemplo.
        Cursos disponíveis (formato Nome|ID):
        {product_choices}
        """
        return [system_prompt, user_prompt]

    def _get_specialist_prompt(self, customer: Customer, customer_message: str, conversation_history: List[Dict]):
        product = Product.query.get(customer.selected_product_id)
        if not product:
            return self._get_aline_prompt(customer_message, conversation_history)

        specialist_persona = f"""
        Você é {product.specialist_name or 'um especialista'}, consultor(a) do produto '{product.name}'.
        Sua prova social é: "{product.specialist_social_proof or 'Tenho muita experiência para te ajudar a ter resultados nesta área.'}"
        O número de WhatsApp para suporte humano é +5512996443780.
        """

        funnel_instructions = {
            'Specialist_Intro': f"Sua primeira ação. Apresente-se com sua persona e prova social. Em seguida, ofereça o link de depoimentos com um botão `[botão:Ver o que os alunos dizem|{product.testimonials_link}]`. Ao final, atualize o estado para 'Specialist_Freebie'.",
            'Specialist_Freebie': f"Ofereça o link das aulas gratuitas com um botão `[botão:Acessar Aulas Gratuitas|{product.free_group_link}]`. Pergunte se o cliente quer um cupom de desconto para fechar a compra. Atualize o estado para 'Specialist_Coupon'.",
            'Specialist_Coupon': f"Ofereça um cupom de 50 reais ('50TAO'), diga que é válido por apenas 10 minutos para criar urgência, e envie o link de pagamento com o botão `[botão:Comprar Agora com Desconto|{product.payment_link}]`. Atualize o estado para 'Specialist_Followup'.",
            'Specialist_Followup': "Pergunte se o cliente conseguiu finalizar a compra. Ofereça as opções `[choice:Sim, comprei!]` e `[choice:Não, tive um problema]`.",
            'Specialist_Success': "Dê os parabéns pela compra, informe que o acesso chegará por e-mail e dê um reforço positivo. Finalize a conversa. Atualize o estado para 'Completed'.",
            'Specialist_Problem': f"Peça desculpas pelo problema e envie o link do WhatsApp para o suporte humano com um botão `[botão:Falar com Suporte Humano|https://wa.me/5512996443780]`. Finalize a conversa. Atualize o estado para 'Completed'."
        }
        
        current_state = customer.funnel_state
        if current_state == 'Specialist_Followup':
            if 'sim' in customer_message.lower() or 'comprei' in customer_message.lower():
                current_state = 'Specialist_Success'
            elif 'não' in customer_message.lower() or 'problema' in customer_message.lower():
                current_state = 'Specialist_Problem'

        current_instruction = funnel_instructions.get(current_state, "Converse com o cliente para tentar vender o produto, seguindo o funil.")

        system_prompt = f"""
        {specialist_persona}
        Siga a instrução para a sua etapa atual do funil: "{current_instruction}"
        Responda APENAS com um JSON contendo "analysis", "response", e opcionalmente "funnel_state_update" com o próximo estado do funil.
        Seja sempre breve e amigável (1-2 frases).
        """
        user_prompt = f"""
        Histórico: {self._build_history_for_gemini(conversation_history)}
        Cliente: "{customer_message}"
        """
        return [system_prompt, user_prompt]

    def _build_history_for_gemini(self, conversation_history: List[Dict], limit: int = 6) -> str:
        if not conversation_history: return "Início da conversa."
        return "\n".join([f"{'Cliente' if m['message_type'] == 'incoming' else 'Assistente'}: {m['message_content']}" for m in conversation_history[-limit:]])
