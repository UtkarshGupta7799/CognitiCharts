import tensorflow as tf

def make_tf_model(seq_len, num_features, num_classes=3):
    inputs = tf.keras.Input(shape=(seq_len, num_features))
    x = tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
