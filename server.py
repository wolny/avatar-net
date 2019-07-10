import io
import time

import numpy as np
import scipy
import tensorflow as tf
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
from flask import send_file

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
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)


def imsave(filename, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(filename, quality=95)


class StyleTransfer(Resource):
    def get_np_images(self, request, keys):
        imgs = []
        for key in keys:
            content = request.files[key]
            bytes = io.BytesIO()
            content.save(bytes)
            np_img = np.array(Image.open(bytes))
            if len(np_img.shape) == 2:
                np_img = np.dstack((np_img, np_img, np_img))
            if np_img.shape[2] == 4:
                np_img = np_img[:, :, :3]
            imgs.append(np_img)
        return imgs

    def post(self):
        imgs = self.get_np_images(request, ['content', 'style'])
        np_content_img = imgs[0]
        np_style_img = imgs[1]

        start_time = time.time()
        np_stylized_image = sess.run(stylized_image,
                                     feed_dict={inp_content_image: np_content_img,
                                                inp_style_image: np_style_img})
        duration = time.time() - start_time
        print("---%s seconds ---" % duration)

        imsave('tmp.jpg', np_stylized_image)
        return send_file('tmp.jpg')


api.add_resource(StyleTransfer, '/transfer')

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=8888)
