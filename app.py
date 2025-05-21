from flask import Flask, request, render_template
from model_runner import model_runner

app = Flask(__name__)
model = model_runner()

@app.route("/", methods=["GET", "POST"])
def index():
    reply = ""
    if request.method == "POST":
        user_input = request.form["message"]
        reply = model.infer(user_input)  # ← pass plain text directly
    return render_template("index.html", reply=reply)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

