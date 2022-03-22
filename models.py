from tensorflow.keras import layers, Model

def refiner_model(input_shape):

    def resnet_block(b_input, n_features, kernel_size):

        x = layers.Conv2D(
            n_features,
            kernel_size = kernel_size,
            activation = 'relu',
            padding = 'same') (b_input)

        x = layers.Conv2D(
            n_features,
            kernel_size = kernel_size,
            padding = 'same') (x)

        x = layers.add([x, b_input])

        x = layers.Activation('relu') (x)

        return x

    input_l = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu') (input_l)

    for _ in range(4): x = resnet_block(x, 64, kernel_size=3)

    # GAN Training hacks: https://github.com/soumith/ganhacks recommends tanh
    output_l = layers.Conv2D(input_shape[-1], kernel_size=1, padding='same', activation='tanh') (x)

    return Model(inputs=input_l, outputs=output_l, name="R")

def descriminator_model(input_shape):

    input_l = layers.Input(shape=input_shape)

    x = layers.Conv2D(
            96,
            kernel_size = 3,
            strides = 2,
            padding = 'same',
            activation = 'relu') (input_l)

    x = layers.Conv2D(
            64,
            kernel_size = 3,
            strides = 2,
            padding = 'same',
            activation = 'relu') (x)

    x = layers.MaxPooling2D(
            pool_size = 3,
            strides = 1,
            padding = 'same') (x)

    x = layers.Conv2D(
            32,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            activation = 'relu') (x)

    x = layers.Conv2D(
            32,
            kernel_size = 1,
            strides = 1,
            padding = 'same',
            activation = 'relu') (x)

    x = layers.Conv2D(
            2,
            kernel_size = 1,
            strides = 1,
            padding = 'same',
            activation = 'relu') (x)

    output_l = layers.Reshape(target_shape=(-1, 2)) (x)

    return Model(inputs=input_l, outputs=output_l, name="D")

if __name__ == "__main__":
    INPUT_SHAPE = (35, 55, 1)
    R = refiner_model(INPUT_SHAPE)
    D = descriminator_model(INPUT_SHAPE)

    R.summary()
    print()
    D.summary()