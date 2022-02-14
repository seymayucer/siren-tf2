import tensorflow as tf

import glob
import argparse
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path
from siren import Siren
from data import get_img


def train_siren(image_path, batch_size, num_epochs, output_dir, img_size):
    img_mask, img_train, _ = get_img(image_path, img_size=img_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((img_mask, img_train))
    train_dataset = train_dataset.shuffle(256).batch(batch_size).cache()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = Siren(
        units=256,
        out_features=3,
        num_layers=3,
        hidden_omega=30.0,
        outermost_linear=True,
    )

    batch_size = min(batch_size, len(img_mask))
    num_steps = int(len(img_mask) * num_epochs / batch_size)
    print("Total training steps : ", num_steps)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.MeanSquaredError()  # Sum of squared error
    model.compile(optimizer, loss=loss)

    checkpoint_dir = os.path.join(output_dir, "checkpoints/")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    logdir = os.path.join(output_dir, "logs/", timestamp)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_dir + "model",
            monitor="loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        ),
        tf.keras.callbacks.TensorBoard(logdir, update_freq="batch", profile_batch=20),
    ]

    model.fit(train_dataset, epochs=num_epochs, callbacks=callbacks, verbose=2)


def eval(test_img_path, batch_size, output_dir, img_size):
    img_mask, img_eval, img_ground_truth = get_img(test_img_path, img_size=img_size)

    # Build model
    model = Siren(
        units=256,
        out_features=3,
        num_layers=3,
        hidden_omega=30.0,
        outermost_linear=True,
    )
    _ = model(tf.zeros([1, 2]))

    # Restore model
    checkpoint_dir = os.path.join(output_dir, "checkpoints/model")
    if len(glob.glob(checkpoint_dir + "*.index")) == 0:
        raise FileNotFoundError("Model checkpoint not found !")

    # instantiate model
    test_dataset = tf.data.Dataset.from_tensor_slices((img_mask, img_eval))
    test_dataset = test_dataset.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    # load checkpoint
    model.load_weights(checkpoint_dir).expect_partial()  # skip optimizer loading

    predicted_image = model.predict(test_dataset, batch_size=batch_size, verbose=1)

    predicted_image = predicted_image.reshape(img_ground_truth.shape)
    predicted_image = predicted_image.clip(0.0, 1.0)

    fig, axes = plt.subplots(1, 2)
    plt.sca(axes[0])
    plt.imshow(img_ground_truth.numpy())
    plt.title("Ground Truth Image", color="#767676")
    plt.axis("off")

    plt.sca(axes[1])
    plt.imshow(predicted_image)
    plt.title("Predicted Image", color="#767676")
    plt.axis("off")

    fig.tight_layout()
    output_image_path = f"{output_dir}/{Path(test_img_path).stem}.png"
    plt.savefig(
        output_image_path,
        dpi=150,
        transparent=True,
    )


def main():

    parser = argparse.ArgumentParser("SIREN Train and Test Script for Image Fitting")

    parser.add_argument("--input_image", type=str, default="samples/leaves.jpg")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="it stores checkpoints and logs and generated images",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--n_epochs", type=int, default=5, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256 * 256,
        help="batch_size of training",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="batch_size of training",
    )
    opt = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if opt.train:
        train_siren(
            opt.input_image, opt.batch_size, opt.n_epochs, opt.output_dir, opt.img_size
        )

    eval(opt.input_image, opt.batch_size, opt.output_dir, opt.img_size)


if __name__ == "__main__":
    main()
