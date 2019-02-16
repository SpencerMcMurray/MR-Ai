from flask import Flask, render_template, request, redirect, url_for
from run_img import run_img_thru_model
import os
from functions import read_counter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024**3  # 1GB

MODEL_PATH = "output/unet_model_for_decathlon.hdf5"
OUTPUT_PATH = "inference_examples"


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_path = request.files['img']
        if img_path is None:
            redirect(url_for('index'))
        img_id = read_counter('id_counter')
        run_img_thru_model(img_path, MODEL_PATH, OUTPUT_PATH, img_id)
        brain = os.path.join(OUTPUT_PATH, "brain" + str(img_id) + ".png")
        tumor = os.path.join(OUTPUT_PATH, "pred" + str(img_id) + ".png")
        return render_template('scansDisplay.html', brain=brain, tumor=tumor)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
