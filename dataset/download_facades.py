import tensorflow as tf
import os
from pathlib import Path

def main():
    _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    BASE_PATH = Path(__file__)

    path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                          origin=_URL,
                                          extract=True,
                                          cache_dir=str(BASE_PATH.parent),
                                          cache_subdir='')

    # PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

main()
