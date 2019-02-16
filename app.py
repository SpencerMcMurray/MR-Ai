from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Run Script
        brain = "Yeet"
        tumor = "Yaah"
        return render_template('scansDisplay.html', brain=brain, tumor=tumor)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
