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
            return {
                "analysis": {"intent": "api_error", "sentiment": 0.0},
                "response": "Desculpe, tive um problema para processar sua solicitação. Poderia tentar de novo?"
            }

    def generate_response(self, customer: Customer, conversation_history: List[Dict]) -> Dict[str, Any]:
        default_error_response = {
            "analysis": {"intent": "internal_error", "sentiment": 0.0},
            "response": "Desculpe, ocorreu um erro interno. Por favor, tente novamente."
        }
        
        try:
            # O Agente agora sempre atua como especialista.
            prompt_parts = self._get_specialist_prompt(customer, conversation_history)
            structured_response = self._make_api_call(prompt_parts)

            if structured_response and 'analysis' in structured_response and 'response' in structured_response:
                logger.info(f"Resposta gerada para estado '{customer.funnel_state}': {structured_response.get('analysis')}")
                return structured_response
            
            logger.error(f"A API do Gemini retornou um JSON inválido. Resposta: {structured_response}")
            return default_error_response
            
        except Exception as e:
            logger.error(f"Erro em generate_response: {e}", exc_info=True)
            return default_error_response

    def _get_specialist_prompt(self, customer: Customer, conversation_history: List[Dict]):
        last_message = conversation_history[-1]['message_content'] if conversation_history else ''
        product = Product.query.get(customer.selected_product_id)
        
        if not product:
            return ["Responda apenas com este JSON: {\"analysis\":{\"intent\":\"error\"}, \"response\":\"Erro: produto não encontrado.\"}"]

        specialist_persona = f"""
        Você é {product.specialist_name or 'um especialista'}, consultor(a) do produto '{product.name}'.
        Sua prova social é: "{product.specialist_social_proof or 'Tenho muita experiência para te ajudar a ter resultados nesta área.'}"
        O número de WhatsApp para suporte humano é +5512996443780.
        """

        funnel_instructions = {
            'Specialist_Intro': (
                f"Sua primeira ação. Apresente-se com sua persona e prova social. "
                f"Sua resposta deve ter duas partes separadas por '%%DELAY%%'. "
                f"Primeiro, ofereça o link das aulas gratuitas com o botão `[botão:Acessar Aulas Gratuitas|{product.free_group_link}]`. "
                f"Depois do '%%DELAY%%', ofereça o link de depoimentos com o botão `[botão:Ver o que os alunos dizem|{product.testimonials_link}]`. "
                f"Finalmente, ofereça as opções `[choice:Gostei, quero saber mais!]` e `[choice:Tenho uma dúvida]`. "
                f"Atualize o estado para 'Specialist_Offer'."
            ),
            'Specialist_Offer': (
                f"O cliente quer continuar. Ofereça o cupom '50TAO' (válido por 10 minutos) e o link de pagamento. "
                f"Formate sua resposta exatamente assim: primeiro o texto, depois o cupom em uma nova linha, depois o aviso de validade e por fim o botão de pagamento `[botão:Comprar Agora com Desconto|{product.payment_link}]`. "
                f"Ofereça as opções `[choice:Consegui comprar!]` e `[choice:Tive um problema]`. "
                f"Atualize o estado para 'Specialist_Followup'."
            ),
            'Specialist_Followup': ( # Este estado agora é o ponto de partida para o follow-up
                "O cliente indicou que teve um problema ou não comprou. "
                "Inicie um atendimento mais livre e persuasivo. Seu objetivo é entender a principal objeção (preço? tempo? confiança?) e tentar contorná-la com base nos benefícios do produto. "
                "Seja empático. No final, ofereça as opções `[choice:Entendi, vou comprar agora]` e `[choice:Ainda não sei...]`."
                "Se ele escolher comprar, atualize o estado para 'Specialist_Offer'. Se não, para 'Specialist_FinalAttempt'."
            ),
            'Specialist_FinalAttempt': (
                f"Esta é a última tentativa. O cliente ainda está indeciso. "
                f"Diga que entende a dúvida dele, mas que a oportunidade é única. "
                f"Ofereça o botão para falar com o suporte humano no WhatsApp: `[botão:Falar com Suporte Humano|https://wa.me/5512996443780]`. "
                f"Atualize o estado para 'Completed'."
            ),
            'Specialist_Success': "Dê os parabéns pela compra, informe que o acesso chegará por e-mail e dê um reforço positivo. Finalize a conversa. Atualize o estado para 'Completed'."
        }
        
        current_state = customer.funnel_state
        # Lógica para tratar a resposta do cliente e decidir o próximo passo
        if current_state == 'Specialist_Offer':
            # Se o cliente demonstrou interesse após a intro, vai para a oferta.
            pass # A instrução já é a correta
        elif 'problema' in last_message.lower() or 'não' in last_message.lower():
            current_state = 'Specialist_Followup'
        elif 'comprei' in last_message.lower() or 'sim' in last_message.lower():
            current_state = 'Specialist_Success'

        current_instruction = funnel_instructions.get(current_state, funnel_instructions['Specialist_Intro'])

        system_prompt = f"""
        {specialist_persona}
        Siga a instrução para a sua etapa atual do funil: "{current_instruction}"
        REGRA CRÍTICA: Sua resposta DEVE SER um único objeto JSON válido, sem markdown.
        O JSON deve conter "analysis" (com "intent" e "sentiment"), "response" (a mensagem para o cliente), e "funnel_state_update" (com o próximo estado do funil).
        """
        user_prompt = f"""
        Histórico: {self._build_history_for_gemini(conversation_history)}
        Cliente: "{last_message}"
        """
        return [system_prompt, user_prompt]

    def _build_history_for_gemini(self, conversation_history: List[Dict], limit: int = 6) -> str:
        if not conversation_history: return "Início da conversa."
        return "\n".join([f"{'Cliente' if m['message_type'] == 'incoming' else 'Assistente'}: {m['message_content']}" for m in conversation_history[-limit:]])
