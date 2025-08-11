import os
import json
import logging
from typing import Dict, List, Any, Optional

# --- ALTERAÇÃO 1: Importando a biblioteca do Google ---
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from models import Product

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        # --- ALTERAÇÃO 2: Configurando o cliente do Google Gemini ---
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("A chave da API do Google não está configurada.")
            # Lançar um erro impede que a aplicação inicie sem a chave, o que é mais seguro.
            raise ValueError("GOOGLE_API_KEY não encontrada no ambiente.")
        
        genai.configure(api_key=api_key)
        
        # Usando o gemini-1.5-flash, que é rápido e tem um ótimo plano gratuito.
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        
        # Configurações para forçar a saída em JSON e definir segurança
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

    # --- ALTERAÇÃO 3: A função de chamada de API foi totalmente reescrita para o Google SDK ---
    def _make_api_call(self, prompt_parts: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Envia o prompt para a API do Gemini e retorna a resposta em JSON.
        """
        try:
            response = self.model.generate_content(
                prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            response_text = response.text
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado na chamada da API do Gemini: {e}")
            logger.error(f"Resposta recebida do Gemini (pode conter o erro): {getattr(e, 'response', 'N/A')}")
            return None

    def generate_response(self, customer_message: str, conversation_history: List[Dict], available_products: List[Product]) -> Dict[str, Any]:
        default_error_response = {
            "analysis": {"intent": "error", "sentiment": 0.0},
            "response": "Desculpe, estou com um problema técnico no momento. Pode tentar novamente em alguns minutos?"
        }
        if not available_products:
            default_error_response["response"] = "Olá! No momento, estamos atualizando nosso catálogo."
            return default_error_response

        try:
            product_context = self._build_product_context(available_products)
            system_prompt = self._create_system_prompt()
            
            # --- ALTERAÇÃO 4: A forma de montar o histórico e o prompt foi adaptada para o Gemini ---
            history_for_gemini = self._build_history_for_gemini(conversation_history)
            
            # O prompt para o Gemini é uma lista de "partes" que formam a conversa
            final_prompt = [
                system_prompt,
                *history_for_gemini,
                "---",
                "PRODUTOS DISPONÍVEIS PARA VOCÊ OFERECER:",
                product_context,
                "---",
                "MENSAGEM ATUAL DO CLIENTE PARA VOCÊ RESPONDER:",
                f"Cliente: \"{customer_message}\""
            ]

            structured_response = self._make_api_call(final_prompt)

            if structured_response and 'analysis' in structured_response and 'response' in structured_response:
                logger.info(f"Análise e resposta geradas com sucesso via Gemini: {structured_response['analysis']}")
                return structured_response
            
            logger.error(f"A API do Gemini retornou um JSON inválido ou com estrutura incorreta. Resposta: {structured_response}")
            return default_error_response
            
        except Exception as e:
            logger.error(f"Ocorreu um erro inesperado em generate_response: {e}")
            return default_error_response

    def _build_product_context(self, available_products: List[Product]) -> str:
        product_context_lines = []
        for p in available_products:
            try:
                benefits_list = json.loads(p.key_benefits)
                benefits_str = ", ".join(benefits_list)
            except:
                benefits_str = "N/A"

            price_info = f"R${p.price:.2f}"
            if p.original_price and p.original_price > p.price:
                price_info = f"de R${p.original_price:.2f} por {price_info}"
            
            product_line = (
                f"- Produto: {p.name} | Benefícios: {benefits_str} | Preço: {price_info} | "
                f"Link Pag.: {p.payment_link or 'N/A'} | Link Grátis: {p.free_group_link or 'N/A'}"
            )
            product_context_lines.append(product_line)
        return "\n".join(product_context_lines)

    def _build_history_for_gemini(self, conversation_history: List[Dict], limit: int = 6) -> List[str]:
        if not conversation_history:
            return []
        
        history_lines = []
        for msg in conversation_history[-limit:]:
            role = "Cliente" if msg.get('message_type') == 'incoming' else "Aline"
            history_lines.append(f"{role}: {msg.get('message_content', '')}")
        return history_lines

    def _create_system_prompt(self) -> str:
        # Este prompt foi refinado para ser claro para o modelo Gemini
        system_prompt = f"""
        Você é Aline, uma vendedora especialista em produtos digitais. Sua tarefa é analisar a mensagem do cliente e formular a próxima resposta.
        Siga estritamente as seguintes regras:
        1.  **PERSONA:** Seu tom é amigável e direto, como em uma conversa de WhatsApp (1-2 frases).
        2.  **FORMATO DA SAÍDA:** Sua resposta DEVE SER um único objeto JSON válido. Não inclua markdown (```json```).
        3.  **ESTRUTURA JSON:** O JSON deve conter exatamente duas chaves: "analysis" e "response".
            - "analysis": um objeto com sua análise. Deve conter a chave "intent" (com um dos seguintes valores: 'interesse_inicial', 'duvida_produto', 'objecao_preco', 'pronto_para_comprar', 'desinteressado', 'saudacao', 'pedindo_link_gratuito', 'pedindo_link_pagamento') e a chave "sentiment" (um número de -1.0 a 1.0).
            - "response": uma string com a resposta da Aline para o cliente. Use os princípios de venda abaixo para formular esta resposta.
        4.  **PRINCÍPIOS DE VENDA:**
            - Use os "Benefícios Principais" do produto para convencer.
            - Se o cliente perguntar o preço, responda com a informação de "Preço".
            - Para enviar um link, use o formato `[botão:Texto do Botão|URL]`.
            - Para dar opções, use o formato `[choice:Texto da Opção]`, cada um em uma nova linha.

        EXEMPLO DE SAÍDA JSON VÁLIDA:
        {{
          "analysis": {{
            "intent": "interesse_inicial",
            "sentiment": 0.5
          }},
          "response": "Olá! Que bom te ver por aqui. Me diga o que você busca:\\n[choice:Quero saber os preços]\\n[choice:Ver todos os cursos]"
        }}
        """
        return system_prompt.strip()
