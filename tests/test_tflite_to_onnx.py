import zipfile
from pathlib import Path

import numpy as np
import pytest

from pose_craft.estimation import tflite_to_onnx
from pose_craft.estimation.tflite_to_onnx import densify


class _Meta:
    """Duck-typed stand-in for tflite.DimensionMetadata."""

    def __init__(self, dense_size=None, segments=None, indices=None):
        self._dense_size, self._segments, self._indices = dense_size, segments, indices

    def Format(self):
        return 0 if self._dense_size is not None else 1

    def DenseSize(self):
        return self._dense_size

    def ArraySegmentsType(self):
        return 1

    def ArraySegments(self):
        return self._segments

    def ArrayIndicesType(self):
        return 1

    def ArrayIndices(self):
        return self._indices


class _Sparsity:
    """Duck-typed stand-in for tflite.SparsityParameters."""

    def __init__(self, traversal_order, dim_metadata, block_map=()):
        self._order, self._meta, self._block_map = traversal_order, dim_metadata, block_map

    def TraversalOrderAsNumpy(self):
        return np.array(self._order)

    def BlockMapAsNumpy(self):
        return np.array(self._block_map)

    def BlockMapLength(self):
        return len(self._block_map)

    def DimMetadataLength(self):
        return len(self._meta)

    def DimMetadata(self, i):
        return self._meta[i]


@pytest.fixture
def plain_index_vectors(monkeypatch):
    # The fakes hand back their index arrays directly instead of flatbuffer tables.
    monkeypatch.setattr(tflite_to_onnx, "_sparse_index_vector", lambda vector_type, table: np.asarray(table))


def test_densify_csr_matrix(plain_index_vectors):
    dense = np.array([[0, 5, 0, 0], [0, 0, 0, 0], [7, 0, 0, 9]], dtype=np.float32)
    sparsity = _Sparsity(
        traversal_order=[0, 1],
        dim_metadata=[_Meta(dense_size=3), _Meta(segments=[0, 1, 1, 3], indices=[1, 0, 3])],
    )
    values = np.array([5, 7, 9], dtype=np.float32)
    np.testing.assert_array_equal(densify(values, [3, 4], sparsity), dense)


def test_densify_block_sparse_matrix(plain_index_vectors):
    # 4x4 matrix stored as 2x2 blocks: only block (row 0, col 1) and block (1, 0) are non-zero.
    dense = np.zeros((4, 4), dtype=np.float32)
    dense[0:2, 2:4] = [[1, 2], [3, 4]]
    dense[2:4, 0:2] = [[5, 6], [7, 8]]
    sparsity = _Sparsity(
        traversal_order=[0, 1, 2, 3],
        block_map=[0, 1],
        dim_metadata=[
            _Meta(dense_size=2),
            _Meta(segments=[0, 1, 2], indices=[1, 0]),
            _Meta(dense_size=2),
            _Meta(dense_size=2),
        ],
    )
    values = np.arange(1, 9, dtype=np.float32)
    np.testing.assert_array_equal(densify(values, [4, 4], sparsity), dense)


def test_densify_rejects_a_value_count_that_does_not_match_the_metadata(plain_index_vectors):
    sparsity = _Sparsity(traversal_order=[0, 1], dim_metadata=[_Meta(dense_size=2), _Meta(dense_size=2)])
    with pytest.raises(ValueError):
        densify(np.zeros(3, dtype=np.float32), [2, 2], sparsity)


_CACHED_TASK = Path.home() / ".cache" / "pose_craft" / "models" / "pose_landmarker_lite.task"


@pytest.mark.skipif(not _CACHED_TASK.exists(), reason="needs the downloaded lite pose model")
def test_converted_models_match_the_tflite_reference_runtime():
    """End-to-end converter check against LiteRT (TFLite's own interpreter), when
    both it and a downloaded model are available -- a developer-machine test; CI
    has neither and skips it."""
    interpreter_module = pytest.importorskip("ai_edge_litert.interpreter")
    ort = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(_CACHED_TASK) as bundle:
        for member in tflite_to_onnx.TASK_BUNDLE_MEMBERS.values():
            data = bundle.read(member)
            session = ort.InferenceSession(
                tflite_to_onnx.convert_tflite_bytes(data).SerializeToString(), providers=["CPUExecutionProvider"]
            )
            interpreter = interpreter_module.Interpreter(model_content=data)
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            x = rng.random(input_detail["shape"], dtype=np.float32)
            interpreter.set_tensor(input_detail["index"], x)
            interpreter.invoke()
            expected = {d["name"]: interpreter.get_tensor(d["index"]) for d in interpreter.get_output_details()}
            actual = dict(zip([o.name for o in session.get_outputs()], session.run(None, {input_detail["name"]: x})))
            for name, value in actual.items():
                # Tolerance relative to each output's own range, with an absolute floor
                # for near-zero outputs (e.g. the pose-presence score on a random input).
                atol = max(1e-3 * float(np.abs(expected[name]).max()), 1e-5)
                np.testing.assert_allclose(value, expected[name], rtol=1e-3, atol=atol)


# --- B11: the hand landmarker's extra ops (PRELU, MEAN, FULLY_CONNECTED) ----------
#
# Checked against LiteRT on MediaPipe's real hand_landmarker.task (palm detector
# and 21-point landmark model): worst relative error 2e-6 / 4e-6. These tests pin
# the layout handling (NHWC model, NCHW graph) on one-op graphs, with numpy as the
# reference, without needing LiteRT or a model download.


class _StubTensor:
    def __init__(self, name, shape):
        self._name, self._shape = name, shape

    def Name(self):
        return self._name.encode()

    def ShapeAsNumpy(self):
        return np.array(self._shape)


class _StubOp:
    def __init__(self, code, inputs, outputs):
        self._code, self._inputs, self._outputs = code, inputs, outputs

    def OpcodeIndex(self):
        return self._code

    def InputsAsNumpy(self):
        return np.array(self._inputs)

    def OutputsAsNumpy(self):
        return np.array(self._outputs)


class _StubCode:
    def __init__(self, code):
        self._code = code

    def BuiltinCode(self):
        return self._code

    def DeprecatedBuiltinCode(self):
        return 0


class _StubOptions:
    def __init__(self, keep_dims=False):
        self._keep = keep_dims

    def KeepDims(self):
        return self._keep

    def WeightsFormat(self):
        return 0

    def FusedActivationFunction(self):
        return 0


def _one_op_model(op_name, input_shape, output_shape, constants, monkeypatch, keep_dims=False):
    """A converted ONNX model for: input (tensor 0) -> op -> output (tensor 1),
    with `constants` as tensors 2, 3, ... (the op's other inputs, in order)."""
    pytest.importorskip("onnx")
    import tflite

    code = getattr(tflite.BuiltinOperator, op_name)
    shapes = [input_shape, output_shape] + [list(c.shape) for c in constants]
    tensors = [_StubTensor(f"t{i}", s) for i, s in enumerate(shapes)]
    op = _StubOp(0, [0] + [2 + i for i in range(len(constants))], [1])

    class _Graph:
        def InputsLength(self): return 1
        def Inputs(self, i): return 0
        def OutputsLength(self): return 1
        def Outputs(self, i): return 1
        def OperatorsLength(self): return 1
        def Operators(self, i): return op
        def Tensors(self, i): return tensors[i]

    class _Model:
        def OperatorCodes(self, i): return _StubCode(code)

    converter = tflite_to_onnx._Converter.__new__(tflite_to_onnx._Converter)
    converter._tflite = tflite
    converter.model, converter.graph = _Model(), _Graph()
    converter.op_names = {code: op_name}
    converter.nodes, converter.initializers, converter.values = [], [], {}
    converter.constants = {2 + i: c for i, c in enumerate(constants)}
    converter._name_counter = 0
    converter._read_constant = lambda index: None
    monkeypatch.setattr(tflite_to_onnx, "_options", lambda op, cls: _StubOptions(keep_dims))
    return converter.convert()


def _run(model, x):
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"t0": x})[0]


def test_prelu_slope_broadcasts_over_nhwc_channels(monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 4, 5, 3)).astype(np.float32)
    slope = np.array([[[0.1, 0.5, 2.0]]], dtype=np.float32)  # [1, 1, C], as the palm detector stores it
    model = _one_op_model("PRELU", [1, 4, 5, 3], [1, 4, 5, 3], [slope], monkeypatch)
    np.testing.assert_allclose(_run(model, x), np.where(x >= 0, x, x * slope), rtol=1e-6)


@pytest.mark.parametrize("keep_dims", [False, True])
def test_mean_over_the_spatial_axes(monkeypatch, keep_dims):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 4, 5, 6)).astype(np.float32)
    out_shape = [1, 1, 1, 6] if keep_dims else [1, 6]
    model = _one_op_model("MEAN", [1, 4, 5, 6], out_shape, [np.array([1, 2], dtype=np.int32)], monkeypatch, keep_dims)
    np.testing.assert_allclose(_run(model, x), x.mean(axis=(1, 2), keepdims=keep_dims), rtol=1e-5, atol=1e-6)


def test_fully_connected_flattens_in_nhwc_order(monkeypatch):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    weights = rng.normal(size=(5, 24)).astype(np.float32)  # [out, in]
    bias = rng.normal(size=(5,)).astype(np.float32)
    model = _one_op_model("FULLY_CONNECTED", [1, 2, 3, 4], [1, 5], [weights, bias], monkeypatch)
    np.testing.assert_allclose(_run(model, x), x.reshape(1, -1) @ weights.T + bias, rtol=1e-5, atol=1e-5)
