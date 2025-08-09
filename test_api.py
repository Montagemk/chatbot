import requests

CHATBOT_URL = 'https://atpchatbot.onrender.com/webhook'

data = {
    "sender": "teste_script_py",
    "message": "Olá, qual o preço do curso?"
}

response = requests.post(CHATBOT_URL, json=data)

if response.status_code == 200:
    print("Resposta da IA:")
    print(response.json())
else:
    print(f"Erro na requisição: {response.status_code}")
    print(response.text)
