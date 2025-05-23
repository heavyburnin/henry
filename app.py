from flask import Flask, request, render_template, jsonify, session, Response, stream_with_context
from model_runner import model_runner
from concurrent.futures import ThreadPoolExecutor
from flask_session import Session

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Replace with a secure key
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

model = model_runner()
executor = ThreadPoolExecutor(max_workers=4)

@app.route("/chat", methods=["GET"])
def chat_get():
    history = session.get("chat_history", [])
    return render_template("index.html", history=history)

@app.route("/chat", methods=["POST"])
def chat_post():
    # Handle both form and AJAX POSTs
    user_input = ""
    if request.is_json:
        data = request.get_json()
        user_input = data.get("message", "")
    else:
        user_input = request.form.get("message", "")

    if not user_input.strip():
        return jsonify({"reply": "Please enter a message."}) if request.is_json else render_template("index.html")

    future = executor.submit(model.infer, user_input)
    reply = future.result()

    # Store chat history in session
    history = session.get("chat_history", [])
    history.append({"user": user_input, "bot": reply})
    session["chat_history"] = history

    if request.is_json:
        return jsonify({"reply": reply})
    else:
        return render_template("index.html", history=history)

@app.route("/chat/stream", methods=["GET"])
def chat_stream():
    user_input = request.args.get("message", "")
    if not user_input:
        return "Missing message", 400

    def generate():
        for chunk in model.infer_stream(user_input):
            yield f"data: {chunk}\n\n"
        yield "event: end\ndata: end\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route("/chat/api", methods=["POST"])
def chat_api():
    if request.is_json:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in JSON"}), 400
        user_input = data["message"]
        future = executor.submit(model.infer, user_input)
        reply = future.result()
        return jsonify({"response": reply})
    return jsonify({"error": "Expected application/json POST"}), 400

# switched to gunicorn
#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

