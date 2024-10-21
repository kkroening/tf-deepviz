import pathlib
import tensorflow as tf
import tfdeepviz

_ROOT_PATH = svg_path = pathlib.Path(__file__).parent.parent
_HELLO_WORLD_SVG_PATH = _ROOT_PATH / 'doc' / 'hello-world.svg'


def test__tf_hello_world_example():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    actual = tfdeepviz.to_svg(model)
    # _HELLO_WORLD_SVG_PATH.write_text(actual, 'utf-8')
    expected = _HELLO_WORLD_SVG_PATH.read_text('utf-8')
    assert actual == expected
