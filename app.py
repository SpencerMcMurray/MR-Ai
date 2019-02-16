from flask import Flask, render_template, request, redirect, url_for
from run_img import run_img_thru_model
import os
from functions import read_counter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024**3  # 1GB

MODEL_PATH = "single-node/output/unet_model_for_decathlon.hdf5"
OUTPUT_PATH = "static/images/scans"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan', methods=["POST"])
def scan():
    img = request.files['img']
    if img is None:
        redirect(url_for('index'))
    img_id = read_counter('id_counter')
    img_path = os.path.join(OUTPUT_PATH, "temp" + str(img_id) + ".nii")
    img.save(img_path)

    run_img_thru_model(img_path, MODEL_PATH, OUTPUT_PATH, img_id)
    brain = "brain" + str(img_id) + ".png"
    tumor = "pred" + str(img_id) + ".png"
    os.remove(img_path)
    return render_template('scansDisplay.html', brain=brain, tumor=tumor)


if __name__ == "__main__":
    app.run()
