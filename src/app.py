import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo salvo
modelo = joblib.load('modelo_regressao_logistica.pkl')

# Carregar a base de dados
df = pd.read_csv('creditcard.csv')

# Interface Streamlit
st.title('Previsão com Modelo ML')

# Exemplo de input do usuário para características do modelo
feature1 = st.number_input('Feature 1:', value=0.0)
feature2 = st.number_input('Feature 2:', value=0.0)
# ... adicione inputs para todas as características do seu modelo

# Processamento dos dados para previsão
dados_usuario = [[feature1, feature2]]  # Coloque as características em uma lista

# Realizar a previsão com o modelo
resultado = modelo.predict(dados_usuario)

# Exibir o resultado
st.write('Resultado da Previsão:')
st.write(resultado)
