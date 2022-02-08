import tensorflow as tf


def get_train_img(img_path, sampling_ratio):
    img_raw = tf.io.read_file(img_path)
    img_ground_truth = tf.io.decode_image(img_raw, channels=3, dtype=tf.float32)

    rows, cols, channels = img_ground_truth.shape
    pixel_count = rows * cols
    sampled_pixel_count = int(pixel_count * sampling_ratio)

    img_mask_x = tf.random.uniform(
        [sampled_pixel_count], maxval=rows, seed=0, dtype=tf.int32
    )
    img_mask_y = tf.random.uniform(
        [sampled_pixel_count], maxval=cols, seed=1, dtype=tf.int32
    )

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

    img_mask_idx = tf.concat([img_mask_x, img_mask_y], axis=-1)
    img_train = tf.gather_nd(img_ground_truth, img_mask_idx, batch_dims=0)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

    img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)

    return img_mask, img_train


def get_test_img(img_path):

    img_raw = tf.io.read_file(img_path)
    img_ground_truth = tf.io.decode_image(img_raw, channels=3, dtype=tf.float32)

    rows, cols, channels = img_ground_truth.shape

    img_mask_x = tf.range(0, rows, dtype=tf.int32)
    img_mask_y = tf.range(0, cols, dtype=tf.int32)

    img_mask_x, img_mask_y = tf.meshgrid(img_mask_x, img_mask_y, indexing="ij")

    img_mask_x = tf.expand_dims(img_mask_x, axis=-1)
    img_mask_y = tf.expand_dims(img_mask_y, axis=-1)

    img_mask_x = tf.cast(img_mask_x, tf.float32) / rows
    img_mask_y = tf.cast(img_mask_y, tf.float32) / cols

    img_mask = tf.concat([img_mask_x, img_mask_y], axis=-1)
    img_mask = tf.reshape(img_mask, [-1, 2])

    img_train = tf.reshape(img_ground_truth, [-1, 3])

    return img_mask, img_train, img_ground_truth.shape, img_ground_truth
