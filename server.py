import io
import time

import numpy as np
import scipy
import tensorflow as tf
from urllib.parse import urlparse
import secrets
import io
from pathlib import Path
import base64
from urllib.request import urlopen
from flask import Flask, request
from flask_cors import CORS
from PIL import Image

from models import models_factory
from models import preprocessing

slim = tf.contrib.slim

# define required args
tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'tmp/tfmodel',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
# choose the model configuration file
tf.app.flags.DEFINE_string(
    'model_config_path', None,
    'The path of the model configuration file.')
tf.app.flags.DEFINE_float(
    'inter_weight', 1.0,
    'The blending weight of the style patterns in the stylized image')
FLAGS = tf.app.flags.FLAGS

# define computational graph
if not FLAGS.checkpoint_dir:
    raise ValueError('You must supply the checkpoints directory with '
                     '--checkpoint_dir')

tf.logging.set_verbosity(tf.logging.INFO)
graph = tf.get_default_graph()

with graph.as_default():
    # define the model
    style_model, options = models_factory.get_model(FLAGS.model_config_path)

    # predict the stylized image
    inp_content_image = tf.placeholder(tf.float32, shape=(None, None, 3))
    inp_style_image = tf.placeholder(tf.float32, shape=(None, None, 3))

    # preprocess the content and style images
    content_image = preprocessing.mean_image_subtraction(inp_content_image)
    content_image = tf.expand_dims(content_image, axis=0)
    # style resizing and cropping
    style_image = preprocessing.preprocessing_image(
        inp_style_image,
        448,
        448,
        style_model.style_size)
    style_image = tf.expand_dims(style_image, axis=0)

    # style transfer
    stylized_image = style_model.transfer_styles(content_image, style_image, inter_weight=FLAGS.inter_weight)
    stylized_image = tf.squeeze(stylized_image, axis=0)

    # starting inference of the images
    init_fn = slim.assign_from_checkpoint_fn(FLAGS.checkpoint_dir, slim.get_model_variables(), ignore_missing_vars=True)
    sess = tf.Session(graph=graph)
    init_fn(sess)



app = Flask(__name__)
CORS(app)


OUTPUT_DIR = Path("output")
STYLE_IMG_DIR = Path("images")


def save_and_b64encode(arr, suffix, email):
    root = OUTPUT_DIR / email
    root.mkdir(mode=0o755, parents=True, exist_ok=True)

    f = io.BytesIO()
    Image.fromarray(arr).save(f, format="jpeg")

    uid = secrets.token_hex(nbytes=16)
    buf = f.getbuffer()
    (root / f"{uid}-{suffix}.jpg").write_bytes(buf)

    return base64.b64encode(buf).decode()


def dataurl2ndarray(dataurl):
    resp = urlopen(dataurl)
    return np.array(Image.open(resp.file))


def get_style_image(url):
    url = urlparse(url)
    path = url.path
    parts = path.strip("/").split("/")
    assert len(parts) == 2
    assert parts[0] == "images"
    name = parts[-1]
    
    np_img = np.array(Image.open(STYLE_IMG_DIR / name))

    if len(np_img.shape) == 2:
        np_img = np.dstack((np_img, np_img, np_img))
    if np_img.shape[2] == 4:
        np_img = np_img[:, :, :3]

    return np_img


@app.route("/transfer", methods=["POST"])
def transfer_json():
    json = request.get_json()
    email = json["email"]

    content_image = dataurl2ndarray(json["contentImage"])
    style_image = get_style_image(json["styleImage"])

    stylized_image = sess.run(stylized_image,
                                     feed_dict={inp_content_image: content_image,
                                                inp_style_image: style_image})

    inverse_stylized_image = sess.run(stylized_image,
                                     feed_dict={inp_content_image: style_image,
                                                inp_style_image: content_image})

    image = save_and_b64encode(stylized_image, "image", email)
    inverse_image = save_and_b64encode(inverse_stylized_image, "inverse", email)

    return {
        "image": image,
        "inverseImage": inverse_image,
    }


if __name__ == "__main__":
    app.run()
