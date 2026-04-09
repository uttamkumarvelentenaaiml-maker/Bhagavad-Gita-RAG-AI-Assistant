from flask import Flask, request, jsonify, render_template
from rag import ask

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_api():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"response": "Enter a question"})

    answer = ask(query)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)