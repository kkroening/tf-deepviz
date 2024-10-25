import pathlib
import pytest
import tensorflow as tf
import tfdeepviz

_ROOT_PATH = svg_path = pathlib.Path(__file__).parent.parent
_DOC_PATH = _ROOT_PATH / 'doc'


@pytest.fixture(autouse=True)
def reset_keras():
    """Resets the Keras session, to ensure determinism between tests runs.

    Without this, global counters increment statefully between tests - e.g. layers are
    named ``flatten_3_1``, ``flatten_4_1``, ... etc. rather than ``flatten_1``.

    `<https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session>`_.
    """
    tf.keras.utils.clear_session(free_memory=False)


@pytest.mark.parametrize('minimal', [False, True])
def test__to_svg__example_hello_world(minimal):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    options = tfdeepviz.RenderOptions(minimal=minimal)
    actual = tfdeepviz.to_svg(model, options=options)

    path = _DOC_PATH / ('hello-world-minimal.svg' if minimal else 'hello-world.svg')
    path.write_text(actual, 'utf-8')

    expected = path.read_text('utf-8')
    assert actual == expected


@pytest.mark.parametrize('minimal', [False, True])
def test__to_svg__example_cnn(minimal):
    # model = tf.keras.models.Sequential(
    #     [
    #         tf.keras.layers.Input((28, 28)),
    #         tf.keras.layers.Reshape((28, 28, 1)),
    #         tf.keras.layers.Conv2D(
    #             32,
    #             (3, 3),
    #             activation='relu',
    #             name='conv1',
    #             padding='same',
    #         ),
    #         tf.keras.layers.MaxPooling2D((2, 2)),
    #         tf.keras.layers.Conv2D(
    #             64,
    #             (3, 3),
    #             activation='relu',
    #             name='conv2',
    #             padding='same',
    #         ),
    #         tf.keras.layers.MaxPooling2D((2, 2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(
    #             128,
    #             activation='relu',
    #             name='dense1',
    #         ),
    #         tf.keras.layers.Dropout(0.2),
    #         tf.keras.layers.Dense(10),
    #     ]
    # )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((32, 32, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        ]
    )

    options = tfdeepviz.RenderOptions(minimal=minimal)
    actual = tfdeepviz.to_svg(model, options=options)

    path = _DOC_PATH / ('cnn-example-minimal.svg' if minimal else 'cnn-example.svg')
    path.write_text(actual, 'utf-8')

    expected = path.read_text('utf-8')
    assert actual == expected
