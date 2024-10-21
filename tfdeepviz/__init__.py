# pylint: disable=no-member
import graphviz  # type: ignore
import itertools
import pathlib
import tensorflow  # type: ignore
from collections.abc import Iterable
from typing import overload

_NODE_ATTRS = dict(
    style='filled',
    shape='box',
    align='left',
    fillcolor='darkseagreen2',
    fontsize='10',
    ranksep='0.1',
    height='0.2',
    fontname='monospace',
)

_SUBGRAPH_ATTRS = dict(
    penwidth='2',
    fontname='monospace',
    fontsize='12',
    labelloc='b',
)


def _identify_node(
    prefix: str,
    op: tensorflow.Operation,
) -> tuple[str, bool]:
    parts = op.name.removeprefix(prefix).split('/', 1)
    node_label = parts[0]
    is_cluster = len(parts) > 1
    return node_label, is_cluster


def _render_subgraph(
    digraph: graphviz.Digraph,
    parent_ops: Iterable[tensorflow.Operation],
    prefix: str,
) -> None:
    ops = sorted(
        (x for x in parent_ops if x.name.startswith(prefix)), key=lambda x: x.name
    )

    groups = itertools.groupby(ops, lambda x: _identify_node(prefix, x))
    for (node_label, is_cluster), node_ops in groups:
        if is_cluster:
            cluster_name = prefix + node_label
            with digraph.subgraph(name=f'cluster_{cluster_name}') as subgraph:
                subgraph.attr(
                    'graph',
                    label=node_label,
                    **_SUBGRAPH_ATTRS,
                )
                _render_subgraph(subgraph, node_ops, cluster_name + '/')
        else:
            for node_op in node_ops:
                digraph.node(node_op.name, label=node_label, **_NODE_ATTRS)
                for node_input in node_op.inputs:
                    input_name = node_input.name.lstrip('^').split(':')[0]
                    digraph.edge(input_name, node_op.name)


def _tf_graph_to_digraph(
    tf_graph: tensorflow.Graph,
) -> graphviz.Digraph:
    """Converts a TensorFlow graph object into a hierarchical Graphviz Digraph."""
    digraph = graphviz.Digraph(
        format='svg',
        engine='dot',
        graph_attr=dict(
            rankdir='TB',
            nodesep='0.4',
            ranksep='0.4',
        ),
        node_attr=_NODE_ATTRS,
    )
    _render_subgraph(digraph, tf_graph.operations, '')
    return digraph


def _generate_dummy_inputs(
    model: tensorflow.keras.Model,
) -> list[tensorflow.Tensor]:
    """
    Automatically generates placeholder/dummy tf.constant tensors based on the
    model's input shapes.
    """
    dummy_inputs = []
    for input_spec in model.inputs:
        # Fill any shape dimensions having `None` placeholders with a default of 1.
        # For example, it's common to have the batch size parameter as `None` as a
        # placeholder - i.e. `(None, 28, 28)` -> `(1, 28, 28)`.
        input_shape = tuple(x if x is not None else 1 for x in input_spec.shape)

        dummy_input = tensorflow.constant(
            0.0, shape=input_shape, dtype=input_spec.dtype
        )
        dummy_inputs.append(dummy_input)

    return dummy_inputs


def trace_model_graph(
    model: tensorflow.keras.Model,
    inputs: list[tensorflow.Tensor] | None = None,
) -> tensorflow.Graph:
    if inputs is None:
        inputs = _generate_dummy_inputs(model)

    @tensorflow.function
    def model_fn(*model_inputs: tensorflow.Tensor) -> tensorflow.Tensor:
        return model(*model_inputs)

    return model_fn.get_concrete_function(*inputs).graph


def to_digraph(
    model_or_graph: tensorflow.keras.Model | tensorflow.Graph,
    inputs: list[tensorflow.Tensor] | None = None,
) -> graphviz.Digraph:
    """Converts a Keras model or a TensorFlow graph into a hierarchical Graphviz Digraph.

    Args:
    - model_or_graph: A TensorFlow Keras model or a TensorFlow graph object.
    - inputs: List of input tensors to feed to the model. If None, auto-generates dummy
      inputs for a Keras model.

    Returns:
    - A Graphviz Digraph representing the model or graph structure.
    """
    # If the input is a Keras model, we need to trace it to generate a graph
    if isinstance(model_or_graph, tensorflow.keras.Model):
        graph = trace_model_graph(model_or_graph, inputs)
    elif isinstance(model_or_graph, tensorflow.Graph):
        graph = model_or_graph
    else:
        raise TypeError('Input must be a Keras model or TensorFlow graph.')

    # Pass the graph to the renderer (assumed to be already implemented)
    return _tf_graph_to_digraph(graph)


@overload
def to_svg(
    model_or_graph: tensorflow.keras.Model | tensorflow.Graph,
    filename: str,
    inputs: list[tensorflow.Tensor] | None = None,
) -> None: ...


@overload
def to_svg(
    model_or_graph: tensorflow.keras.Model | tensorflow.Graph,
    filename: None,
    inputs: list[tensorflow.Tensor] | None = None,
) -> str: ...


def to_svg(
    model_or_graph: tensorflow.keras.Model | tensorflow.Graph,
    filename: pathlib.Path | str | None = None,
    inputs: list[tensorflow.Tensor] | None = None,
) -> str | None:
    digraph = to_digraph(model_or_graph, inputs)
    svg_data = digraph.pipe(format='svg').decode('utf-8')
    if filename is not None:
        path = pathlib.Path(filename)
        path.write_text(svg_data, 'utf-8')
        result = None
    else:
        result = svg_data
    return result


__all__ = ['to_svg']
