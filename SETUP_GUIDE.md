Com certeza\! Para evitar qualquer confusão e ter um guia que reflita exatamente o seu projeto atual, aqui está o arquivo `SETUP_GUIDE.md` refeito e completo.

Este novo arquivo remove todas as menções à integração do WhatsApp e foca na sua nova estratégia de usar o chatbot no site e implantá-lo no Render.

### **Arquivo `SETUP_GUIDE.md` (Completo e Refeito)**

```
# IA de Vendas para Web Chat

## Visão Geral

Esta é uma aplicação web inteligente que usa IA e aprendizado por reforço para conduzir conversas de vendas de forma autônoma. O sistema se integra a um site via uma API REST para interagir com clientes em português, usando o modelo Meta Llama 3.1 8B Instruct do OpenRouter para gerar respostas dinâmicas e contextuais. A IA aprende com cada interação, otimizando estratégias de vendas com base em conversões bem-sucedidas para maximizar a receita de vendas de produtos digitais.

## Arquitetura do Sistema

### Arquitetura do Web Framework
- **Aplicação web baseada em Flask** com `SQLAlchemy ORM` para operações de banco de dados.
- **Rotas centralizadas** no `routes.py` para uma separação clara de preocupações.
- **Frontend baseado em templates** usando `Jinja2` e `Bootstrap` para uma interface responsiva.
- **Endpoints de API RESTful** para lidar com webhooks e recuperar dados.

### Arquitetura do Agente de IA
- **Integração com a API do OpenRouter** usando o modelo Meta Llama 3.1 8B Instruct para geração de linguagem natural e análise da intenção do cliente.
- **Sistema de classificação de intenção** que categoriza as mensagens dos clientes em intenções predefinidas.
- **Geração de resposta dinâmica** com base no contexto do cliente, histórico de conversa e informações do produto.

### Sistema de Aprendizado por Reforço
- **Abordagem "multi-armed bandit"** com quatro estratégias de vendas: `consultivo`, `escassez`, `emocional`, `racional`.
- **Equilíbrio entre exploração e explotação** com uma taxa de exploração de 20% e fator de decaimento de 95%.
- **Rastreamento de sucesso** baseado em conversões de vendas reais como sinais de recompensa.
- **Otimização de estratégia** que aprende quais abordagens funcionam melhor para diferentes tipos e contextos de clientes.

### Esquema do Banco de Dados
- **Tabela Customer** para rastrear números de telefone, histórico de interação e status de compra.
- **Tabela Conversation** para armazenar todas as mensagens com metadados da estratégia de IA e pontuações de sentimento.
- **Tabela Sale** para registrar conversões bem-sucedidas com rastreamento de receita.
- **Tabela AILearningData** para manter métricas de desempenho de estratégia e taxas de sucesso.

### Análise e Monitoramento
- **Painel em tempo real** exibindo taxas de conversão, métricas de receita e estatísticas de clientes.
- **Rastreamento de desempenho de estratégia** mostrando taxas de sucesso para diferentes abordagens de IA.
- **Visualizador de histórico de conversas** para revisão manual e garantia de qualidade.
- **Estatísticas de aprendizado** para monitorar a melhoria da IA ao longo do tempo.

## Dependências Externas

### Serviços de IA
- **API do OpenRouter** para geração de linguagem natural e análise de resposta usando o modelo Meta Llama 3.1 8B Instruct (modelo gratuito).
- Requer a variável de ambiente `OPENROUTER_API_KEY`.

### Banco de Dados
- **Banco de dados compatível com `SQLAlchemy`** (padrão `SQLite`, configurável via `DATABASE_URL`).

### Bibliotecas Python
- **Framework web Flask** com `SQLAlchemy` para operações de banco de dados.
- **Biblioteca Requests** para chamadas de API HTTP para o OpenRouter.
- **Werkzeug ProxyFix** para tratamento adequado de cabeçalhos em implantações de produção.
- **Flask-Cors** para permitir a comunicação com seu site.

## Implantação e Integração

### Configuração da Aplicação
- A configuração da aplicação é baseada em variáveis de ambiente para chaves de API, conexões de banco de dados e informações do produto.
- Variáveis de ambiente necessárias: `OPENROUTER_API_KEY` e `DATABASE_URL`.

### Processo de Implantação
- A aplicação está pronta para ser implantada no Render.
- Para implantar, envie o seu código para o GitHub, conecte o Render ao seu repositório e configure as variáveis de ambiente.
- O banco de dados é criado automaticamente na primeira execução através de um script de inicialização.

### Integração do Chatbot no Site
- O chatbot se comunica com o seu site através do endpoint `/webhook`.
- O seu JavaScript deve enviar mensagens via `POST` para `https://sua-aplicacao-no-render.com/webhook`.
- O seu servidor irá processar a mensagem e retornar uma resposta em JSON, que o seu JavaScript irá exibir na janela de chat.

## Solução de Problemas

### Problemas Comuns
1.  **A IA não está respondendo:**
    * Verifique se a variável de ambiente `OPENROUTER_API_KEY` está definida e se a chave é válida.
    * Verifique os logs do Render para ver se há erros.
    
2.  **Erros no banco de dados:**
    * Certifique-se de que a variável de ambiente `DATABASE_URL` está correta.
    * Verifique os logs para garantir que a migração do banco de dados (`db.create_all()`) foi bem-sucedida.

### Recursos de Suporte
- Documentação do Render: `https://render.com/docs`
- Documentação do OpenRouter: `https://openrouter.ai/docs`
```