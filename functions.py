import os
from run_img import run_img_thru_model


MODEL_PATH = "single-node/output/unet_model_for_decathlon.hdf5"
OUTPUT_PATH = "static/images/scans"


def counter(filepath):
    """(str) -> int
    Takes in the filepath of a file with one integer in it, then returns that integer, then increments it.
    """
    file = open(filepath, "r")
    curr_id = int(file.readline().strip())
    file.close()

    file = open(filepath, "w")
    file.write(str(curr_id + 1))
    file.close()

    return curr_id


def create_images(img, img_id):
    """(FileStorage, int) -> str, str
    Uses the image given to create two new files representing itself and the AI-rendered tumor detection, returning
    the names for both files respectively
    """
    img_path = os.path.join(OUTPUT_PATH, "temp" + str(img_id) + ".nii")
    img.save(img_path)

    run_img_thru_model(img_path, MODEL_PATH, OUTPUT_PATH, img_id)
    brain = "brain" + str(img_id) + ".png"
    tumor = "pred" + str(img_id) + ".png"
    os.remove(img_path)
    # Removes any imgs stored previously to save space
    past_brain, past_tumor = "brain" + str(img_id-1) + ".png", "pred" + str(img_id-1) + ".png"
    if os.path.exists(os.path.join(OUTPUT_PATH, past_brain)):
        os.remove(os.path.join(OUTPUT_PATH, past_brain))
    if os.path.exists(os.path.join(OUTPUT_PATH, past_tumor)):
        os.remove(os.path.join(OUTPUT_PATH, past_tumor))
    return brain, tumor
