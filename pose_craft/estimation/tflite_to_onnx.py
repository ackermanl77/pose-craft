"""Converts the two TFLite graphs inside a MediaPipe pose-landmarker `.task` bundle
into ONNX, so they can run on ONNX Runtime (and therefore on an NVIDIA GPU through
its CUDA execution provider -- MediaPipe's own Windows build is CPU-only).

This is deliberately *not* a general TFLite converter. It supports exactly the op
set those BlazePose graphs use (checked for the lite/full/heavy variants):
CONV_2D, DEPTHWISE_CONV_2D, ADD, PAD, MAX_POOL_2D, RESIZE_BILINEAR,
DEPTH_TO_SPACE, CONCATENATION, RESHAPE and LOGISTIC -- plus PRELU, MEAN and
FULLY_CONNECTED, the extra ops of MediaPipe's hand landmarker (palm detector and
21-point hand landmark model, for finger poses) -- plus DEQUANTIZE and DENSIFY,
which only ever act on constant weights (float16 and block-sparse storage) and so
are folded into plain float32 initializers here. Anything else raises
`UnsupportedTFLiteOp` instead of silently producing a wrong graph -- the general
converters (tf2onnx needs TensorFlow, which has no Python 3.14 build; tflite2onnx
mishandles float16 and these PAD ops) were tried first and don't work on these files.

Layout: TFLite is NHWC, ONNX's Conv/MaxPool/Resize are NCHW. The ONNX graph keeps
the TFLite model's NHWC input and output shapes (so callers see the same tensors
MediaPipe's own calculators do); internally every rank-4 activation is held as NCHW,
with a Transpose inserted at the graph input, around RESHAPE, and at rank-4 outputs.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

# Opset 17 is supported by every ONNX Runtime release since 1.14.
ONNX_OPSET = 17

# Bump whenever a change here alters the generated graphs: cached conversions are
# named after it (see landmarker.onnx_model_paths), so a bump forces a re-convert.
CONVERTER_VERSION = 1

_NHWC_TO_NCHW = [0, 3, 1, 2]
_NCHW_TO_NHWC = [0, 2, 3, 1]
# Where an NHWC axis ends up once the tensor is stored as NCHW.
_NHWC_AXIS_IN_NCHW = {0: 0, 1: 2, 2: 3, 3: 1}

# tflite.TensorType values -> numpy dtypes (only the ones these graphs use).
_TENSOR_DTYPES = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64, 9: np.int8}

# tflite.ActivationFunctionType values.
_ACT_NONE, _ACT_RELU, _ACT_RELU_N1_TO_1, _ACT_RELU6, _ACT_TANH = 0, 1, 2, 3, 4

# tflite.Padding values.
_PADDING_SAME = 0


class UnsupportedTFLiteOp(ValueError):
    pass


def _builtin_operator_names() -> dict[int, str]:
    from tflite.BuiltinOperator import BuiltinOperator

    return {value: name for name, value in vars(BuiltinOperator).items() if not name.startswith("_")}


def _options(op, options_cls):
    table = op.BuiltinOptions()
    options = options_cls()
    if table is not None:
        options.Init(table.Bytes, table.Pos)
    return options


def _sparse_index_vector(vector_type: int, table) -> np.ndarray:
    import tflite

    classes = {1: tflite.Int32Vector, 2: tflite.Uint16Vector, 3: tflite.Uint8Vector}
    if vector_type not in classes:
        raise UnsupportedTFLiteOp(f"unsupported sparse index vector type {vector_type}")
    vector = classes[vector_type]()
    vector.Init(table.Bytes, table.Pos)
    return vector.ValuesAsNumpy().astype(np.int64)


def densify(values: np.ndarray, dense_shape: list[int], sparsity) -> np.ndarray:
    """Expands a TFLite sparse (CSR / block-sparse) constant into its dense array.

    A port of TFLite's own `FormatConverter::SparseToDense`
    (tensorflow/lite/kernels/internal/utils/sparsity_format_converter.cc): the stored
    values are laid out in `traversal_order`, one entry per dimension level, where a
    level is either dense (every index 0..dense_size-1) or CSR (segments + indices).
    Levels past the tensor's own rank are block dimensions, mapped back to an original
    dimension through `block_map`.
    """
    rank = len(dense_shape)
    traversal_order = [int(v) for v in sparsity.TraversalOrderAsNumpy()]
    block_map = [int(v) for v in sparsity.BlockMapAsNumpy()] if sparsity.BlockMapLength() else []
    levels = []
    for level in range(sparsity.DimMetadataLength()):
        meta = sparsity.DimMetadata(level)
        if meta.Format() == 0:  # DENSE
            levels.append(("dense", int(meta.DenseSize()), None, None))
        else:  # SPARSE_CSR
            segments = _sparse_index_vector(meta.ArraySegmentsType(), meta.ArraySegments())
            indices = _sparse_index_vector(meta.ArrayIndicesType(), meta.ArrayIndices())
            levels.append(("csr", 0, segments, indices))
    block_sizes = [levels[rank + i][1] for i in range(len(block_map))]

    # Walk the level tree breadth-first, carrying (per stored-value path) the index
    # at every level so far plus the "previous index" CSR segments are keyed by --
    # vectorized over all paths at once instead of TFLite's per-element recursion.
    level_indices: list[np.ndarray] = []
    prev = np.zeros(1, dtype=np.int64)
    for kind, dense_size, segments, indices in levels:
        if kind == "dense":
            new_idx = np.tile(np.arange(dense_size, dtype=np.int64), len(prev))
            parent = np.repeat(np.arange(len(prev)), dense_size)
            new_prev = prev[parent] * dense_size + new_idx
        else:
            starts, ends = segments[prev], segments[prev + 1]
            counts = ends - starts
            parent = np.repeat(np.arange(len(prev)), counts)
            positions = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if len(prev) else np.zeros(0, np.int64)
            positions = positions.astype(np.int64)
            new_idx = indices[positions]
            new_prev = positions
        level_indices = [idx[parent] for idx in level_indices] + [new_idx]
        prev = new_prev

    if len(prev) != values.size:
        raise ValueError(f"sparse tensor has {values.size} stored values but its metadata describes {len(prev)}")

    orig = [np.zeros(len(prev), dtype=np.int64) for _ in range(rank)]
    for level in range(rank):
        orig[traversal_order[level]] = level_indices[level]
    for level in range(rank, len(levels)):
        block_idx = traversal_order[level] - rank
        dim = block_map[block_idx]
        orig[dim] = orig[dim] * block_sizes[block_idx] + level_indices[level]

    dense = np.zeros(dense_shape, dtype=values.dtype)
    dense[tuple(orig)] = values.reshape(-1)
    return dense


class _Converter:
    def __init__(self, tflite_bytes: bytes):
        import tflite

        self._tflite = tflite
        self.model = tflite.Model.GetRootAsModel(tflite_bytes, 0)
        if self.model.SubgraphsLength() != 1:
            raise UnsupportedTFLiteOp("only single-subgraph TFLite models are supported")
        self.graph = self.model.Subgraphs(0)
        self.op_names = _builtin_operator_names()
        self.nodes = []
        self.initializers = []
        self.values: dict[int, str] = {}  # tflite tensor index -> ONNX value name
        self.constants: dict[int, np.ndarray] = {}  # tflite tensor index -> folded constant (TFLite layout)
        self._name_counter = 0

    # -- tensors --------------------------------------------------------------

    def _tensor(self, index: int):
        return self.graph.Tensors(index)

    def _shape(self, index: int) -> list[int]:
        return [int(v) for v in self._tensor(index).ShapeAsNumpy()]

    def _unique(self, base: str) -> str:
        self._name_counter += 1
        return f"{base}__{self._name_counter}"

    def _read_constant(self, index: int) -> np.ndarray | None:
        tensor = self._tensor(index)
        buffer = self.model.Buffers(tensor.Buffer())
        if buffer is None or buffer.DataLength() == 0:
            return None
        dtype = _TENSOR_DTYPES.get(tensor.Type())
        if dtype is None:
            raise UnsupportedTFLiteOp(f"unsupported tensor type {tensor.Type()} for {tensor.Name().decode()}")
        raw = buffer.DataAsNumpy().tobytes()
        values = np.frombuffer(raw, dtype=dtype).copy()
        sparsity = tensor.Sparsity()
        if sparsity is not None:
            # Stays sparse-encoded until its DENSIFY op folds it.
            return values
        return values.reshape(self._shape(index))

    def _constant(self, index: int) -> np.ndarray | None:
        if index in self.constants:
            return self.constants[index]
        value = self._read_constant(index)
        if value is not None:
            self.constants[index] = value
        return value

    def _initializer(self, array: np.ndarray, base: str) -> str:
        from onnx import numpy_helper

        name = self._unique(base)
        self.initializers.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def _value(self, index: int) -> str:
        """ONNX value name for a TFLite tensor, materializing a constant as an
        initializer (converted to the NCHW convention if it's rank-4)."""
        if index in self.values:
            return self.values[index]
        constant = self._constant(index)
        if constant is None:
            raise UnsupportedTFLiteOp(f"tensor {self._tensor(index).Name().decode()} has no producer")
        array = constant.astype(np.float32) if constant.dtype == np.float16 else constant
        if array.ndim == 4:
            array = array.transpose(_NHWC_TO_NCHW)
        name = self._initializer(array, self._tensor(index).Name().decode())
        self.values[index] = name
        return name

    def _node(self, op_type: str, inputs: list[str], base: str, **attrs) -> str:
        from onnx import helper

        output = self._unique(base)
        self.nodes.append(helper.make_node(op_type, inputs, [output], name=output, **attrs))
        return output

    def _activation(self, value: str, activation: int, base: str) -> str:
        if activation == _ACT_NONE:
            return value
        if activation == _ACT_RELU:
            return self._node("Relu", [value], base + "_relu")
        if activation in (_ACT_RELU6, _ACT_RELU_N1_TO_1):
            low, high = (0.0, 6.0) if activation == _ACT_RELU6 else (-1.0, 1.0)
            lo = self._initializer(np.array(low, dtype=np.float32), base + "_min")
            hi = self._initializer(np.array(high, dtype=np.float32), base + "_max")
            return self._node("Clip", [value, lo, hi], base + "_clip")
        if activation == _ACT_TANH:
            return self._node("Tanh", [value], base + "_tanh")
        raise UnsupportedTFLiteOp(f"unsupported fused activation {activation}")

    # -- ops ------------------------------------------------------------------

    def _conv_attrs(self, options, kernel_hw: tuple[int, int]) -> dict:
        attrs = {
            "kernel_shape": list(kernel_hw),
            "strides": [options.StrideH(), options.StrideW()],
        }
        if hasattr(options, "DilationHFactor"):
            attrs["dilations"] = [options.DilationHFactor() or 1, options.DilationWFactor() or 1]
        # TFLite's SAME puts any odd extra padding at the end, which is ONNX's SAME_UPPER.
        attrs["auto_pad"] = "SAME_UPPER" if options.Padding() == _PADDING_SAME else "VALID"
        return attrs

    def convert_op(self, op) -> None:
        tfl = self._tflite
        code_entry = self.model.OperatorCodes(op.OpcodeIndex())
        code = max(code_entry.BuiltinCode(), code_entry.DeprecatedBuiltinCode())
        name = self.op_names.get(code, str(code))
        inputs = [int(i) for i in op.InputsAsNumpy()]
        outputs = [int(i) for i in op.OutputsAsNumpy()]
        out_index = outputs[0]
        base = self._tensor(out_index).Name().decode()

        if name == "DEQUANTIZE":
            constant = self._constant(inputs[0])
            if constant is None:
                raise UnsupportedTFLiteOp("DEQUANTIZE of a non-constant tensor")
            quant = self._tensor(inputs[0]).Quantization()
            if constant.dtype == np.float16:
                result = constant.astype(np.float32)
            elif quant is not None and quant.ScaleLength():
                scale = quant.ScaleAsNumpy().astype(np.float32)
                zero_point = quant.ZeroPointAsNumpy().astype(np.float32) if quant.ZeroPointLength() else 0.0
                axis = quant.QuantizedDimension()
                if scale.size > 1:
                    shape = [1] * constant.ndim
                    shape[axis] = -1
                    scale = scale.reshape(shape)
                    if np.ndim(zero_point):
                        zero_point = zero_point.reshape(shape)
                result = (constant.astype(np.float32) - zero_point) * scale
            else:
                result = constant.astype(np.float32)
            self.constants[out_index] = result.reshape(self._shape(out_index))
            return

        if name == "DENSIFY":
            sparsity = self._tensor(inputs[0]).Sparsity()
            constant = self._constant(inputs[0])
            if constant is None or sparsity is None:
                raise UnsupportedTFLiteOp("DENSIFY of a non-constant or non-sparse tensor")
            self.constants[out_index] = densify(constant, self._shape(out_index), sparsity)
            return

        if name == "CONV_2D":
            options = _options(op, tfl.Conv2DOptions)
            weights = self._constant(inputs[1])
            if weights is None:
                raise UnsupportedTFLiteOp("CONV_2D with non-constant weights")
            weights = weights.astype(np.float32).transpose(0, 3, 1, 2)  # OHWI -> OIHW
            conv_inputs = [self._value(inputs[0]), self._initializer(weights, base + "_w")]
            if len(inputs) > 2 and inputs[2] >= 0:
                conv_inputs.append(self._initializer(self._constant(inputs[2]).astype(np.float32), base + "_b"))
            out = self._node("Conv", conv_inputs, base, **self._conv_attrs(options, weights.shape[2:]))
            self.values[out_index] = self._activation(out, options.FusedActivationFunction(), base)
            return

        if name == "DEPTHWISE_CONV_2D":
            options = _options(op, tfl.DepthwiseConv2DOptions)
            weights = self._constant(inputs[1])
            if weights is None:
                raise UnsupportedTFLiteOp("DEPTHWISE_CONV_2D with non-constant weights")
            in_channels = self._shape(inputs[0])[3]
            weights = weights.astype(np.float32).transpose(3, 0, 1, 2)  # [1,H,W,C*m] -> [C*m,1,H,W]
            conv_inputs = [self._value(inputs[0]), self._initializer(weights, base + "_w")]
            if len(inputs) > 2 and inputs[2] >= 0:
                conv_inputs.append(self._initializer(self._constant(inputs[2]).astype(np.float32), base + "_b"))
            attrs = self._conv_attrs(options, weights.shape[2:])
            out = self._node("Conv", conv_inputs, base, group=in_channels, **attrs)
            self.values[out_index] = self._activation(out, options.FusedActivationFunction(), base)
            return

        if name == "ADD":
            options = _options(op, tfl.AddOptions)
            operands = []
            for index in inputs:
                constant = self._constant(index) if index not in self.values else None
                if constant is not None and constant.ndim == 1 and len(self._shape(out_index)) == 4:
                    # A per-channel constant broadcasts over NHWC's last axis; in NCHW
                    # that's axis 1.
                    operands.append(self._initializer(constant.astype(np.float32).reshape(1, -1, 1, 1), base + "_c"))
                else:
                    operands.append(self._value(index))
            out = self._node("Add", operands, base)
            self.values[out_index] = self._activation(out, options.FusedActivationFunction(), base)
            return

        if name == "PAD":
            paddings = self._constant(inputs[1])
            if paddings is None:
                raise UnsupportedTFLiteOp("PAD with non-constant paddings")
            paddings = np.asarray(paddings, dtype=np.int64).reshape(-1, 2)
            if paddings.shape[0] == 4:
                paddings = paddings[_NHWC_TO_NCHW]
            pads = np.concatenate([paddings[:, 0], paddings[:, 1]])
            self.values[out_index] = self._node(
                "Pad", [self._value(inputs[0]), self._initializer(pads, base + "_pads")], base, mode="constant"
            )
            return

        if name == "MAX_POOL_2D":
            options = _options(op, tfl.Pool2DOptions)
            attrs = self._conv_attrs(options, (options.FilterHeight(), options.FilterWidth()))
            out = self._node("MaxPool", [self._value(inputs[0])], base, **attrs)
            self.values[out_index] = self._activation(out, options.FusedActivationFunction(), base)
            return

        if name == "RESIZE_BILINEAR":
            options = _options(op, tfl.ResizeBilinearOptions)
            n, h, w, c = self._shape(out_index)
            if options.AlignCorners():
                mode = "align_corners"
            elif options.HalfPixelCenters():
                mode = "half_pixel"
            else:
                mode = "asymmetric"
            sizes = self._initializer(np.array([n, c, h, w], dtype=np.int64), base + "_sizes")
            self.values[out_index] = self._node(
                "Resize", [self._value(inputs[0]), "", "", sizes], base, mode="linear", coordinate_transformation_mode=mode
            )
            return

        if name == "DEPTH_TO_SPACE":
            options = _options(op, tfl.DepthToSpaceOptions)
            # TensorFlow's DepthToSpace orders the channel dimension as (block_row,
            # block_col, out_channel), which is ONNX's DCR mode.
            self.values[out_index] = self._node(
                "DepthToSpace", [self._value(inputs[0])], base, blocksize=options.BlockSize(), mode="DCR"
            )
            return

        if name == "CONCATENATION":
            options = _options(op, tfl.ConcatenationOptions)
            rank = len(self._shape(out_index))
            axis = options.Axis() % rank
            if rank == 4:
                axis = _NHWC_AXIS_IN_NCHW[axis]
            out = self._node("Concat", [self._value(i) for i in inputs], base, axis=axis)
            self.values[out_index] = self._activation(out, options.FusedActivationFunction(), base)
            return

        if name == "RESHAPE":
            value = self._value(inputs[0])
            if len(self._shape(inputs[0])) == 4:
                # RESHAPE flattens in TFLite's NHWC element order, so undo the NCHW
                # convention first.
                value = self._node("Transpose", [value], base + "_to_nhwc", perm=_NCHW_TO_NHWC)
            out_shape = self._shape(out_index)
            shape = self._initializer(np.array(out_shape, dtype=np.int64), base + "_shape")
            out = self._node("Reshape", [value, shape], base)
            if len(out_shape) == 4:
                out = self._node("Transpose", [out], base + "_to_nchw", perm=_NHWC_TO_NCHW)
            self.values[out_index] = out
            return

        if name == "LOGISTIC":
            self.values[out_index] = self._node("Sigmoid", [self._value(inputs[0])], base)
            return

        if name == "PRELU":
            slope = self._constant(inputs[1])
            if slope is None:
                raise UnsupportedTFLiteOp("PRELU with a non-constant slope")
            slope = slope.astype(np.float32)
            if len(self._shape(inputs[0])) == 4:
                # The slope broadcasts against NHWC's trailing axes (usually
                # [1, 1, C]); pad it to rank 4 and move it to NCHW like the input.
                slope = slope.reshape((1,) * (4 - slope.ndim) + slope.shape).transpose(_NHWC_TO_NCHW)
            self.values[out_index] = self._node(
                "PRelu", [self._value(inputs[0]), self._initializer(slope, base + "_slope")], base
            )
            return

        if name == "MEAN":
            options = _options(op, tfl.ReducerOptions)
            axes = self._constant(inputs[1])
            if axes is None:
                raise UnsupportedTFLiteOp("MEAN with non-constant axes")
            rank = len(self._shape(inputs[0]))
            axes = [int(a) % rank for a in np.atleast_1d(axes)]
            keep_dims = bool(options.KeepDims())
            if rank == 4:
                if not keep_dims and sorted(axes) != [1, 2]:
                    # Only the spatial mean ([N, H, W, C] -> [N, C]) keeps the same
                    # axis order in both layouts once the reduced axes are dropped.
                    raise UnsupportedTFLiteOp(f"MEAN over NHWC axes {axes} without keep_dims")
                axes = [_NHWC_AXIS_IN_NCHW[a] for a in axes]
            self.values[out_index] = self._node(
                "ReduceMean", [self._value(inputs[0])], base, axes=sorted(axes), keepdims=int(keep_dims)
            )
            return

        if name == "FULLY_CONNECTED":
            options = _options(op, tfl.FullyConnectedOptions)
            weights = self._constant(inputs[1])
            if weights is None:
                raise UnsupportedTFLiteOp("FULLY_CONNECTED with non-constant weights")
            if options.WeightsFormat() != 0:
                raise UnsupportedTFLiteOp("FULLY_CONNECTED with shuffled weights")
            weights = weights.astype(np.float32)  # [out, in]
            value = self._value(inputs[0])
            if len(self._shape(inputs[0])) == 4:
                value = self._node("Transpose", [value], base + "_to_nhwc", perm=_NCHW_TO_NHWC)
            if len(self._shape(inputs[0])) != 2:
                flat = self._initializer(np.array([-1, weights.shape[1]], dtype=np.int64), base + "_flat")
                value = self._node("Reshape", [value, flat], base + "_flatten")
            gemm_inputs = [value, self._initializer(weights, base + "_w")]
            if len(inputs) > 2 and inputs[2] >= 0:
                gemm_inputs.append(self._initializer(self._constant(inputs[2]).astype(np.float32), base + "_b"))
            out = self._node("Gemm", gemm_inputs, base, transB=1)
            out_shape = self._shape(out_index)
            if len(out_shape) != 2:
                shape = self._initializer(np.array(out_shape, dtype=np.int64), base + "_shape")
                out = self._node("Reshape", [out, shape], base + "_reshape")
            self.values[out_index] = self._activation(out, options.FusedActivationFunction(), base)
            return

        raise UnsupportedTFLiteOp(f"TFLite op {name} is not supported by this converter")

    def convert(self):
        from onnx import TensorProto, helper

        graph_inputs = []
        for i in range(self.graph.InputsLength()):
            index = int(self.graph.Inputs(i))
            tensor_name = self._tensor(index).Name().decode()
            shape = self._shape(index)
            graph_inputs.append(helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, shape))
            value = tensor_name
            if len(shape) == 4:
                value = self._node("Transpose", [tensor_name], tensor_name + "_nchw", perm=_NHWC_TO_NCHW)
            self.values[index] = value

        for i in range(self.graph.OperatorsLength()):
            self.convert_op(self.graph.Operators(i))

        graph_outputs = []
        for i in range(self.graph.OutputsLength()):
            index = int(self.graph.Outputs(i))
            tensor_name = self._tensor(index).Name().decode()
            shape = self._shape(index)
            value = self._value(index)
            op_type, attrs = ("Transpose", {"perm": _NCHW_TO_NHWC}) if len(shape) == 4 else ("Identity", {})
            self.nodes.append(helper.make_node(op_type, [value], [tensor_name], name=tensor_name, **attrs))
            graph_outputs.append(helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, shape))

        graph = helper.make_graph(self.nodes, "tflite_graph", graph_inputs, graph_outputs, self.initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)], producer_name="pose_craft")
        model.ir_version = 8
        return model


def convert_tflite_bytes(tflite_bytes: bytes):
    """Returns an `onnx.ModelProto` equivalent to the given TFLite flatbuffer."""
    import onnx

    model = _Converter(tflite_bytes).convert()
    onnx.checker.check_model(model)
    return model


TASK_BUNDLE_MEMBERS = {"detector": "pose_detector.tflite", "landmarks": "pose_landmarks_detector.tflite"}
# The hand landmarker bundle (B11): only the 21-point landmark model is used --
# the hand crop comes from the pose's wrist and knuckle landmarks, as in
# MediaPipe Holistic, so the palm detector is never converted.
HAND_TASK_BUNDLE_MEMBERS = {"landmarks": "hand_landmarks_detector.tflite"}


def convert_task_bundle(
    task_path: Path, dest_paths: dict[str, Path], members: dict[str, str] | None = None
) -> dict[str, Path]:
    """Extracts and converts the graphs of a `.task` bundle to `dest_paths`
    (by default both of a pose-landmarker bundle: {"detector": ...,
    "landmarks": ...}; `members` maps each key to its bundle member), returning it.

    Each file is written to a temp name and renamed into place only once complete,
    so an interrupted conversion can never leave behind a file a later run would trust.
    """
    import onnx

    with zipfile.ZipFile(task_path) as bundle:
        for key, member in (TASK_BUNDLE_MEMBERS if members is None else members).items():
            dest = dest_paths[key]
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            try:
                onnx.save_model(convert_tflite_bytes(bundle.read(member)), str(tmp))
                tmp.replace(dest)
            finally:
                tmp.unlink(missing_ok=True)
    return dest_paths
