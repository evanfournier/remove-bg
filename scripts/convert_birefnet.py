"""Makes BiRefNet-lite runnable in the browser, without changing its output.

Usage: python3 scripts/convert_birefnet.py <BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx> <out.onnx>
Requires: pip install onnx numpy

1. Native DeformConv. The public ONNX export emulates torchvision's deform_conv2d
   with GatherND-based bilinear sampling, which creates several 800 MB tensors per
   layer and exceeds the 4 GB WebAssembly heap (std::bad_alloc in the browser).
   Each emulated layer is replaced by one ONNX DeformConv node (opset 19), which
   computes the same function: max difference on the final mask 1e-5, 4x faster.
2. float16 weight storage. Large float32 weights are stored as float16 followed by
   a Cast back to float32, so the file is half the size while every operator still
   computes in float32 (mean mask difference 3e-5).
"""
import collections
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def native_deform_conv(model):
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    graph = model.graph
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.value_info}
    inits = {i.name: i for i in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}
    consumers = collections.defaultdict(list)
    for n in graph.node:
        for i in n.input:
            consumers[i].append(n)

    def const_value(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        node = producer[name]
        assert node.op_type == 'Constant', node.op_type
        return numpy_helper.to_array(node.attribute[0].t)

    scopes = collections.OrderedDict()
    for n in graph.node:
        if '/atrous_conv/' in n.name:
            scopes.setdefault(n.name.split('/atrous_conv/')[0] + '/atrous_conv/', []).append(n)
    assert len(scopes) == 20, f'expected 20 deformable layers, found {len(scopes)}'

    replacements = {}
    for scope, nodes in scopes.items():
        convs = [n for n in nodes if n.op_type == 'Conv']
        offset_conv = [n for n in convs if 'offset_conv' in n.input[1]]
        modulator_conv = [n for n in convs if 'modulator_conv' in n.input[1]]
        final_conv = [n for n in convs if 'regular_conv' in n.input[1]]
        assert len(offset_conv) == len(modulator_conv) == len(final_conv) == 1, scope
        offset_conv, modulator_conv, final_conv = offset_conv[0], modulator_conv[0], final_conv[0]

        x = offset_conv.input[0]
        assert modulator_conv.input[0] == x
        sigmoid = consumers[modulator_conv.output[0]][0]
        mask_mul = consumers[sigmoid.output[0]][0]
        assert sigmoid.op_type == 'Sigmoid' and mask_mul.op_type == 'Mul'
        assert float(const_value([i for i in mask_mul.input if i != sigmoid.output[0]][0])) == 2.0

        weight = final_conv.input[1]
        k = inits[weight].dims[2]
        strides = {a.name: helper.get_attribute_value(a) for a in final_conv.attribute}['strides']
        assert list(strides) == [k, k]
        bias_add = consumers[final_conv.output[0]]
        assert len(bias_add) == 1 and bias_add[0].op_type == 'Add'
        bias_add = bias_add[0]
        bias = const_value([i for i in bias_add.input if i != final_conv.output[0]][0])
        bias_name = scope + 'deform_bias'
        graph.initializer.append(numpy_helper.from_array(bias.reshape(-1).astype(np.float32), bias_name))

        # The export pads x by one extra pixel for its own sampling, so derive the
        # convolution padding from the sizes: out = in + 2p - k + 1.
        h_in, w_in = shapes[x][2:]
        h_out, w_out = shapes[bias_add.output[0]][2:]
        p = (h_out - h_in + k - 1) // 2
        assert h_in + 2 * p - k + 1 == h_out and w_in + 2 * p - k + 1 == w_out, (scope, h_in, h_out, k)
        assert shapes[offset_conv.output[0]] == [1, 2 * k * k, h_out, w_out], scope
        assert shapes[mask_mul.output[0]] == [1, k * k, h_out, w_out], scope

        out = bias_add.output[0]
        replacements[out] = helper.make_node(
            'DeformConv', [x, weight, offset_conv.output[0], bias_name, mask_mul.output[0]], [out],
            name=scope + 'DeformConv', kernel_shape=[k, k], pads=[p, p, p, p], strides=[1, 1],
            dilations=[1, 1], group=1, offset_group=1)

    nodes = [replacements.get(n.output[0], n) if n.output else n for n in graph.node]
    needed = {o.name for o in graph.output}
    keep = []
    for n in reversed(nodes):  # drop everything that no longer feeds the output
        if any(o in needed for o in n.output):
            keep.append(n)
            needed.update(i for i in n.input if i)
    keep.reverse()
    used = [i for i in graph.initializer if i.name in needed]
    del graph.node[:]
    graph.node.extend(keep)
    del graph.initializer[:]
    graph.initializer.extend(used)
    del graph.value_info[:]  # stale intermediate shapes; the runtime re-infers them
    for op in model.opset_import:
        if op.domain in ('', 'ai.onnx'):
            op.version = max(op.version, 19)  # DeformConv
    print(f'native DeformConv: replaced {len(replacements)} layers, {len(keep)} nodes left')
    return model


def float16_storage(model):
    graph = model.graph
    inits, casts = [], []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= 256:
            arr = numpy_helper.to_array(init)
            if np.isfinite(arr).all() and np.abs(arr).max() < 65000:
                half = numpy_helper.from_array(arr.astype(np.float16), init.name + '__fp16')
                inits.append(half)
                casts.append(helper.make_node('Cast', [half.name], [init.name], to=TensorProto.FLOAT,
                                              name=init.name + '__cast'))
                continue
        inits.append(init)

    # Large float Constant nodes: convert only when float16 represents them exactly.
    after = {}
    for node in graph.node:
        if node.op_type != 'Constant':
            continue
        attr = node.attribute[0]
        if attr.name == 'value' and attr.t.data_type == TensorProto.FLOAT and int(np.prod(attr.t.dims)) >= 256:
            arr = numpy_helper.to_array(attr.t)
            half = arr.astype(np.float16)
            if np.array_equal(half.astype(np.float32), arr):
                out = node.output[0]
                attr.t.CopyFrom(numpy_helper.from_array(half, out + '__fp16'))
                node.output[0] = out + '__fp16'
                after[node.output[0]] = helper.make_node('Cast', [node.output[0]], [out],
                                                         to=TensorProto.FLOAT, name=out + '__cast')

    nodes = list(casts)
    for node in graph.node:
        nodes.append(node)
        if node.output and node.output[0] in after:
            nodes.append(after[node.output[0]])
    del graph.initializer[:]
    graph.initializer.extend(inits)
    del graph.node[:]
    graph.node.extend(nodes)
    print(f'float16 storage: {len(casts)} weights, {len(after)} constants')
    return model


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    model = float16_storage(native_deform_conv(onnx.load(src)))
    onnx.checker.check_model(model)
    onnx.save(model, dst)
