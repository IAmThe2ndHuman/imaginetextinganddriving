import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 256x256
image_size = (256, 256)
batch_size = 32


# load images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train_val/",
    validation_split=0.2,
    # color_mode="grayscale",
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train_val/",
    validation_split=0.2,
    # color_mode="grayscale",
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)

# make loading images more efficient
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)
num_classes = 5  # we have 5 categories of images


def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [128, 256, 512]:  # this amount of layers is a good balance between accuracy and generalization
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    activation = "softmax"  # best for multi-class classification
    units = num_classes

    x = layers.Dropout(0.5)(x)  # prevent overfitting
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=(*image_size, 3))
epochs = 50  # usually training will stop before this number of epochs is reached but ehh it doesn't hurt to set a limit

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),  # stop automatically if overfitting is detected
    keras.callbacks.ModelCheckpoint("checkpoints/save_at_{epoch}.h5"),  # save model after every epoch so we can cherry-pick the best iteration
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",  # multi-class classification
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
