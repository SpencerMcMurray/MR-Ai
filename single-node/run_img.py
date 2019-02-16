import matplotlib.pyplot as plt
import keras as K
import nibabel as nib
import numpy as np
import os
import tensorflow as tf
import convert_raw_to_hdf5 as raw_to_hdf5


def combined_dice_ce_loss(y_true, y_pred, axis=(1, 2), smooth=1.,
                          weight=0.9):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return weight * dice_coef_loss(y_true, y_pred, axis, smooth) + \
        (1 - weight) * K.losses.binary_crossentropy(y_true, y_pred)


def dice_coef(y_true, y_pred, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_coef_loss(target, prediction, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.log(2. * numerator) + tf.log(denominator)

    return dice_loss


def load_model_from_path(modelpath):
    """(str) -> Keras Model
    Returns the model at the given path
    """
    return K.models.load_model(modelpath, custom_objects={
        "combined_dice_ce_loss": combined_dice_ce_loss,
        "dice_coef_loss": dice_coef_loss,
        "dice_coef": dice_coef})


def load_img_from_path(imgpath):
    """(str) -> numpy.ndarray
    Returns the image at the given path
    """
    img = np.array(nib.load(imgpath).dataobj)
    if len(img.shape) != 4:  # Make sure 4D
        img = np.expand_dims(img, -1)

    img = raw_to_hdf5.crop_center(img, 144, 144, 4)
    img = raw_to_hdf5.normalize_img(img)

    img = np.swapaxes(np.array(img), 0, -2)

    return img


def run_img_thru_model(imgpath, modelpath, savepath, img_id):
    """(str, str, str, int) -> NoneType
    Runs img through the given trained model and saves the result to savepath
    """
    model = load_model_from_path(modelpath)
    pred = model.predict(load_img_from_path(imgpath))
    K.backend.clear_session()

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
