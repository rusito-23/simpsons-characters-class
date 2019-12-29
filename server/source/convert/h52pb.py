#!/usr/bin/env python
# coding: utf-8

import sys

import tensorflow as tf

if tf.__version__ != "1.14.0":
    sys.exit(f"""
Current tensorflow version not suported, use 1.14.0 instead of {tf.__version__}
    """)

from keras import backend as K
from keras.models import load_model

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def convert_h5_to_pb(h5_path, pb_folder, pb_name):
    K.set_learning_phase(0)

    model = load_model(h5_path)

    frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

    tf.train.write_graph(frozen_graph, pb_folder, pb_name, as_text=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
Convert tensorflow h5 model to pb file.
    """)
    parser.add_argument('--h5_path', '-d', metavar='D', required=True,
                        help='H5 path', dest='h5_path')
    parser.add_argument('--pb_folder', '-o', metavar='D', required=True,
                        help='PB output folder', dest='pb_folder')
    parser.add_argument('--pb_name', '-e', metavar='E', required=True,
                        help='PB file name', dest='pb_name')
    args = parser.parse_args()

    convert_h5_to_pb(args.h5_path, args.pb_folder, args.pb_name)
