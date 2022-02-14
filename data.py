import tensorflow as tf


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [tf.linspace(-1, 1, num=sidelen)])
    mgrid = tf.stack(tf.meshgrid(*tensors, indexing="ij"), axis=-1)
    mgrid = tf.reshape(mgrid, [-1, dim])
    return mgrid


def get_img(img_path, img_size):

    img_raw = tf.io.read_file(img_path)
    img_ground_truth = tf.io.decode_image(img_raw, channels=3, dtype=tf.float32)
    img_ground_truth = tf.image.resize(img_ground_truth, [img_size, img_size])

    return (
        get_mgrid(img_size, 2),
        tf.reshape(img_ground_truth, [img_size * img_size, 3]),
        img_ground_truth,
    )
