import joblib
import pandas as pd
from flask import Flask, request, jsonify
app = Flask(__name__)
try:
    model = joblib.load('modelo_classificacao.joblib')
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'modelo_classificacao.joblib' não encontrado.")
    model = None
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None
@app.route('/prever', methods=['POST'])
def prever_faixa_etaria():
    if model is None:
        return jsonify({'erro': 'Modelo não foi carregado corretamente.'}), 500

    try:
        dados_json = request.get_json()
        
        df_para_prever = pd.DataFrame([dados_json])
        df_para_prever = pd.DataFrame([dados_json])
        previsao = model.predict(df_para_prever)
        resultado = previsao[0]
        return jsonify({'previsao_faixa_etaria': resultado})

    except Exception as e:
        return jsonify({'erro': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)