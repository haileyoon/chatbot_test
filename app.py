from flask import Flask, jsonify, render_template, request
from chat import chatbot_response

app = Flask(__name__)

# Serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint to receive user input and return bot response
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_prompt = data.get("question", "")
    reply = chatbot_response(user_prompt)
    return jsonify(response=reply)

if __name__ == "__main__":
    app.run(debug=True)