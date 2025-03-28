from flask import Flask, render_template, request
from model.feedback_pipeline import feedback_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        keyword = request.form.get("keyword")
        aspects = request.form.get("aspects")
        aspect_list = [aspect.strip() for aspect in aspects.split(",") if aspect.strip()]
        
        results = feedback_pipeline(keyword=keyword, aspects=aspect_list)
        return render_template("results.html", keyword=keyword, results=results)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
