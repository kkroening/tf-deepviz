from __future__ import annotations

# pylint: disable=no-member
import abc
import dataclasses
import graphviz  # type: ignore
import pathlib
import tensorflow  # type: ignore
from dataclasses import dataclass
from typing import Self
from typing import overload

_DEFAULT_EDGE_ATTRS = dict(
    minlen='1.6',
)

_DEFAULT_NODE_ATTRS = dict(
    align='left',
    fontname='monospace',
    fontsize='12',
    height='0.4',
    width='1.0',
    ranksep='0.8',
    shape='box',
    style='filled',
)

_GRAPH_ATTRS = dict(
    rankdir='TB',
    nodesep='0.8',
    ranksep='0.4',
)

_SUBGRAPH_ATTRS = dict(
    penwidth='0.6',
    color='slategray',
    fontname='monospace',
    fontsize='14',
    labelloc='b',
    style='rounded',
)

_YLGNBU9_COLOR_SCHEME = [
    '#f7fcf0',
    '#e0f3db',
    '#ccebc5',
    '#a8ddb5',
    '#7bccc4',
    '#4eb3d3',
    '#2b8cbe',
    '#0868ac',
    '#084081',
]


@dataclass(frozen=True)
class RenderOptions:
    minimal: bool = True


DEFAULT_RENDER_OPTIONS = RenderOptions()


@dataclass(frozen=True)
class Node(abc.ABC):
    name: str
    label: str = dataclasses.field(repr=False)

    @abc.abstractmethod
    def get_descendants(self) -> frozenset[Node]:
        raise NotImplementedError()

    def get_op_map(self) -> dict[str, Op]:
        """Collects all transitively nested (grand-)child :class:`Op` nodes."""
        return {x.name: x for x in self.get_descendants() if isinstance(x, Op)}

    @classmethod
    def from_operations(
        cls,
        operations: list[tensorflow.Operation],
        name: str,
    ) -> Self:
        has_children = next(
            (x for x in operations if x.name.startswith(name) and x.name != name), None
        )
        node = (
            Cluster.from_operations(operations, name)
            if has_children
            else Op.from_operations(operations, name)
        )
        assert isinstance(node, cls)
        return node

    def should_render(
        self,
        options: RenderOptions,  # pylint: disable=unused-argument
    ) -> bool:
        """Determines whether or not this node should be rendered.

        Todo:
            This is a temporary hack/workaround for lack of runtime customizability of
            rendering style.  Subclass(es) hardcode this with highly opinionated rules
            for now.  In the future, this should ideally be replaced by a graph
            transformation approach, where customizing the rendering consists of
            applying one or more graph transformations, such as removing nodes by type,
            collapsing clusters, etc.
        """
        return True

    @abc.abstractmethod
    def render_nodes(
        self,
        digraph: graphviz.Digraph,
        depth: int,
        options: RenderOptions,
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def render_inputs(
        self,
        digraph: graphviz.Digraph,
        op_map: dict[str, Op],
        options: RenderOptions,
    ) -> None:
        raise NotImplementedError()


@dataclass(frozen=True)
class Op(Node):
    operation: tensorflow.Operation

    def get_descendants(self) -> frozenset[Node]:
        return frozenset([self])

    @classmethod
    def from_operations(
        cls,
        operations: list[tensorflow.Operation],
        name: str,
    ) -> Self:
        operation = next(x for x in operations if x.name == name)
        label = name.split('/')[-1] or '.'
        return cls(
            label=label,
            name=name,
            operation=operation,
        )

    def should_render(
        self,
        options: RenderOptions,
    ) -> bool:
        result: bool
        if self.operation.type == 'NoOp':
            result = False
        elif not options.minimal:
            result = True
        else:
            if self.operation.type == 'ReadVariableOp':
                result = False
            elif self.operation.type == 'Const' and self.label in {'shape', 'Shape'}:
                result = False
            else:
                result = True
        return result

    def render_nodes(
        self,
        digraph: graphviz.Digraph,
        depth: int,
        options: RenderOptions,
    ) -> None:
        attrs: dict[str, object] = dict(
            fillcolor='darkseagreen1',
        )
        # attrs |= _DEFAULT_NODE_ATTRS

        if self.operation.type == 'Placeholder':
            attrs.update(
                fillcolor='lightblue',
            )
        elif self.operation.type == 'Const':
            attrs.update(
                fillcolor='whitesmoke',
                style='dashed',
            )
        elif not self.label:
            attrs.update(
                fillcolor='none',
                shape='none',
                height='0.1',
                width='0.8',
                nodesep='4',
            )

        digraph.node(
            self.name,
            label=self.label,
            **attrs,
        )

    def render_inputs(
        self,
        digraph: graphviz.Digraph,
        op_map: dict[str, Op],
        options: RenderOptions,
    ) -> None:
        for input_tensor in list(
            self.operation.inputs
        ):  # + self.operation.control_inputs:
            input_name = input_tensor.name.split(':')[0]
            input_op = op_map.get(input_name)
            if input_op is not None and input_op.should_render(options):
                digraph.edge(input_name, self.name)


@dataclass(frozen=True)
class Cluster(Node):
    children: frozenset[Node] = dataclasses.field(repr=False)

    @property
    def op(self) -> Op | None:
        """The explicit operation node for this cluster, if available.

        Tensorflow graph node clusters are typically implicit. For example, the
        operation name ``sequential_1_1/dense_1_1/MatMul`` implicitly suggests a
        ``sequential_1_1`` even though there's no operation called ``sequential_1_1``.
        In this case, a :class:`Cluster` node is generated implicitly, and :attr:`.op`
        is None since there's no corresponding Tensorflow graph operation.

        But sometimes Tensorflow node clusters are explicit.  For example,
        ``sequential_1_1/flatten_1_1/Reshape/shape`` implies existence of a
        ``sequential_1_1/flatten_1_1/Reshape`` cluster, but there's also an explicit
        operation with the same name as the cluster.  In that case, the :class:`Cluster`
        node's :attr:`.op` points to the cluster's explicit operation.
        """
        return next(
            (x for x in self.children if x.name == self.name and isinstance(x, Op)),
            None,
        )

    def get_descendants(self) -> frozenset[Node]:
        return frozenset(
            descendant
            for child in self.children
            for descendant in child.get_descendants()
        ) | {self}

    @property
    def sorted_children(self) -> list[Node]:
        """The list of children sorted by name.

        Sorting is useful to ensure deterministic Graphviz output.
        """
        return sorted(self.children, key=lambda x: x.name)

    def _get_filtered_children(
        self,
        options: RenderOptions,
    ) -> list[Node]:
        """The list of children filtered for the sake of rendering.

        Todo:
            This is a temporary hack/workaround; see :meth:`Node.should_render`.
        """
        return [x for x in self.sorted_children if x.should_render(options)]

    def should_render(
        self,
        options: RenderOptions,
    ) -> bool:
        """Determines whether or not this node should be rendered."""
        return self.op is None or self.op.should_render(options)

    @classmethod
    def from_operations(
        cls,
        operations: list[tensorflow.Operation],
        name: str,
    ) -> Self:
        child_prefix = name + '/' if name else ''
        child_operations = [x for x in operations if x.name.startswith(child_prefix)]
        child_names = {
            child_prefix + x.name.removeprefix(child_prefix).split('/')[0]
            for x in child_operations
        }
        child_nodes = frozenset(
            Node.from_operations(child_operations, x) for x in sorted(child_names)
        )

        operation = next((x for x in operations if x.name == name), None)
        if operation is not None:
            op_node = Op.from_operations(operations, name)
            op_node = dataclasses.replace(op_node, label='')
            child_nodes = child_nodes | {op_node}

        label = name.split('/')[-1] or '.'
        return cls(
            children=child_nodes,
            label=label,
            name=name,
        )

    def render_nodes(
        self,
        digraph: graphviz.Digraph,
        depth: int,
        options: RenderOptions,
    ) -> None:
        cluster_name = self.name + self.label
        with digraph.subgraph(name=f'cluster_{cluster_name}') as subgraph:
            attrs: dict[str, object] = dict(_SUBGRAPH_ATTRS)

            if self.op is not None:
                attrs |= dict(style='filled')
            bgcolor = _YLGNBU9_COLOR_SCHEME[depth % len(_YLGNBU9_COLOR_SCHEME)]

            subgraph.attr(
                'graph',
                bgcolor=bgcolor,
                fillcolor=bgcolor,
                # penwidth='0.5',
                # style='filled',
                label='    ' + self.label,
                labeljust='r',
                **attrs,
            )
            for child in self._get_filtered_children(options):
                child.render_nodes(subgraph, depth + 1, options)

    def render_inputs(
        self,
        digraph: graphviz.Digraph,
        op_map: dict[str, Op],
        options: RenderOptions,
    ) -> None:
        for child in self._get_filtered_children(options):
            child.render_inputs(digraph, op_map, options)


@dataclass(frozen=True)
class Graph(Cluster):
    @classmethod
    def from_tf_graph(
        cls,
        tf_graph: tensorflow.Graph,
    ) -> Self:
        return cls.from_operations(tf_graph.operations, '')

    def render_nodes(
        self,
        digraph: graphviz.Digraph,
        depth: int,
        options: RenderOptions,
    ) -> None:
        for child in self._get_filtered_children(options):
            child.render_nodes(digraph, depth + 1, options)

    def render(
        self,
        options: RenderOptions = DEFAULT_RENDER_OPTIONS,
    ) -> graphviz.Digraph:
        """Converts a TensorFlow graph object into a hierarchical Graphviz Digraph."""
        digraph = graphviz.Digraph(
            edge_attr=_DEFAULT_EDGE_ATTRS,
            engine='dot',
            format='svg',
            graph_attr=_GRAPH_ATTRS,
            node_attr=_DEFAULT_NODE_ATTRS,
        )
        self.render_nodes(digraph, 0, options)
        op_map = self.get_op_map()
        self.render_inputs(digraph, op_map, options)
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
    options: RenderOptions = DEFAULT_RENDER_OPTIONS,
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
    return Graph.from_tf_graph(graph).render(options)


@overload
def to_svg(
    model_or_graph: tensorflow.keras.Model | tensorflow.Graph,
    filename: str,
    inputs: list[tensorflow.Tensor] | None = None,
    options: RenderOptions = DEFAULT_RENDER_OPTIONS,
) -> None: ...


@overload
def to_svg(
    model_or_graph: tensorflow.keras.Model | tensorflow.Graph,
    filename: None,
    inputs: list[tensorflow.Tensor] | None = None,
    options: RenderOptions = DEFAULT_RENDER_OPTIONS,
) -> str: ...


def to_svg(
    model_or_graph: tensorflow.keras.Model | tensorflow.Graph,
    filename: pathlib.Path | str | None = None,
    inputs: list[tensorflow.Tensor] | None = None,
    options: RenderOptions = DEFAULT_RENDER_OPTIONS,
) -> str | None:
    digraph = to_digraph(model_or_graph, inputs, options)
    svg_data = digraph.pipe(format='svg').decode('utf-8')
    if filename is not None:
        path = pathlib.Path(filename)
        path.write_text(svg_data, 'utf-8')
        result = None
    else:
        result = svg_data
    return result


__all__ = [
    'Cluster',
    'DEFAULT_RENDER_OPTIONS',
    'Graph',
    'Node',
    'Op',
    'RenderOptions',
    'to_digraph',
    'to_svg',
    'trace_model_graph',
]
