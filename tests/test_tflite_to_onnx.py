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
