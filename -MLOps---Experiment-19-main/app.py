import pickle
from flask import Flask,request,jsonify
from prometheus_flask_exporter import PrometheusMetrics

app=Flask(__name__)
metrics=PrometheusMetrics(app)

with open("model.pkl","rb") as file :
    model=pickle.load(file)

@app.route("/home")
def home():
    return jsonify({"message":"app is working.."})

@app.route("/health")
def health():
    return jsonify({"message":"app is healthy"})

@app.route("/predict",methods=["POST"])
def predict():
    
    data = request.get_json()
    s_c = data["source_credibility"]
    a_c = data["author_credibility"]
    c_s = data["clickbait_score"] 
    s_s = data["sentiment_score"]
    a_l = data["article_length"]
    n_q= data["num_quotes"]
    n_i = data["num_images"]
    n_l= data["num_links"]
    s_ss= data["social_shares"]
    v_s= data["verified_score"]
    
    prediction = model.predict([[s_c, a_c, c_s, s_s, a_l, n_q, n_i, n_l, s_ss, v_s]])
    return jsonify({"prediction": prediction[0]})  

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
