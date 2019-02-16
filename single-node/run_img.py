import matplotlib.pyplot as plt
import keras as K
import nibabel as nib
import numpy as np
import os
import plot_inference_examples as plot
import convert_raw_to_hdf5 as raw_To_hdf5


def load_model_from_path(modelpath):
    """(str) -> Keras Model
    Returns the model at the given path
    """
    return K.models.load_model(modelpath, custom_objects={
            "combined_dice_ce_loss": plot.combined_dice_ce_loss,
            "dice_coef_loss": plot.dice_coef_loss,
            "dice_coef": plot.dice_coef})


def load_img_from_path(imgpath):
    """(str) -> numpy.ndarray
    Returns the image at the given path
    """
    img = np.array(nib.load(imgpath).dataobj)
    if len(img.shape) != 4:  # Make sure 4D
        img = np.expand_dims(img, -1)

    img = raw_To_hdf5.crop_center(img, 144, 144, 4)
    img = raw_To_hdf5.normalize_img(img)

    img = np.swapaxes(np.array(img), 0, -2)

    return img


def run_img_thru_model(imgpath, modelpath, savepath, img_id):
    """(str, str, str, int) -> NoneType
    Runs img through the given trained model and saves the result to savepath
    """
    pred = load_model_from_path(modelpath).predict(load_img_from_path(imgpath))
    
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 1, 1)
    plt.imshow(load_img_from_path(imgpath)[0, :, :, 0], cmap="bone", origin="lower")
    plt.axis("off")
    plt.tight_layout()

    name = os.path.join(savepath, "brain{}.png".format(img_id))
    plt.savefig(name, bbox_inches="tight", pad_inches=0)

    plt.clf()
    plt.close()

    plt.figure(figsize=(20, 20))
    
    plt.subplot(1, 1, 1)
    plt.imshow(pred[0, :, :, 0], origin="lower")
    plt.axis("off")
    plt.tight_layout()
    name = os.path.join(savepath, "pred{}.png".format(img_id))
    plt.savefig(name, bbox_inches="tight", pad_inches=0)

    plt.clf()
    plt.close()
