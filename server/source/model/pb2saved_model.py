#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
if tf.__version__ != "1.14.0":
    sys.exit(f"""
Current tensorflow version not suported, use 1.14.0 instead of {tf.__version__}
    """)

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def convert_pb_to_saved_model(export_dir, graph_pb, input_name, output_name):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    with tf.Session(graph=tf.Graph()) as sess:
        # name="" is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        inp = g.get_tensor_by_name(input_name)
        out = g.get_tensor_by_name(output_name)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"in": inp}, {"out": out})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)

    builder.save()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
Convert tensorflow pb model to saved model.
    """)
    parser.add_argument('--export_dir', '-d', metavar='D', required=True,
                        help='Export directory', dest='export_dir')
    parser.add_argument('--pb_file', '-o', metavar='D', required=True,
                        help='PB input file', dest='graph_pb')
    parser.add_argument('--input', '-e', metavar='E', required=True,
                        help='Input name', dest='input_name')
    parser.add_argument('--output', '-e', metavar='E', required=True,
                        help='Output name', dest='output_name')
    args = parser.parse_args()

    convert_pb_to_saved_model(args.export_dir, args.graph_pb, args.input_name, args.output_name)
