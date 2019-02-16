from flask import Flask, render_template, request, redirect, url_for
from functions import counter, create_images

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024**3  # 1GB


@app.route('/')
def index():
    """ The home page """
    return render_template('index.html')


@app.route('/scan', methods=["POST"])
def scan():
    """ The page displaying the results """
    img = request.files['img']
    if img is None:
        redirect(url_for('index'))
    img_id = counter('id_counter')
    brain, tumor = create_images(img, img_id)
    return render_template('scansDisplay.html', brain=brain, tumor=tumor)


if __name__ == "__main__":
    app.run()
