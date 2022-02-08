import tensorflow as tf


class SineInitializer(tf.keras.initializers.Initializer):
    def __init__(self, omega):
        self.omega = omega

    # Keras internally calls supplied initializer as below initializer(shape, dtype=dtype), that is the reason we add dtype here
    def __call__(self, shape, dtype=None):
        limit = tf.sqrt(6 / shape[0]) / self.omega
        return tf.random.uniform(shape, -limit, limit)


class FirstLayerSineInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        limit = 1 / shape[0]
        return tf.random.uniform(shape, -limit, limit)


class SineLayer(tf.keras.layers.Layer):
    def __init__(self, out_features, bias=True, omega_0=30, initializer=None):
        super().__init__()
        self.omega_0 = omega_0

        self.linear = tf.keras.layers.Dense(
            out_features, use_bias=bias, kernel_initializer=initializer
        )

    def call(self, input):
        return tf.sin(self.omega_0 * self.linear(input))

    def call_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return tf.sin(intermediate), intermediate


class Siren(tf.keras.Model):
    def __init__(
        self,
        units,
        num_layers,
        out_features,
        outermost_linear=False,
        hidden_omega=30.0,
    ):
        super().__init__()

        self.net = tf.keras.Sequential()
        self.net.add(
            SineLayer(
                units,
                initializer=FirstLayerSineInitializer(),
            )
        )

        for i in range(num_layers):
            self.net.add(
                SineLayer(
                    units,
                    initializer=SineInitializer(hidden_omega),
                )
            )

        if outermost_linear:
            final_linear = tf.keras.layers.Dense(
                out_features, kernel_initializer=SineInitializer(hidden_omega)
            )
            self.net.add(final_linear)
        else:
            self.net.add(
                SineLayer(
                    out_features,
                    initializer=SineInitializer(hidden_omega),
                )
            )

    def call(self, coords):
        output = self.net(coords)
        return output
