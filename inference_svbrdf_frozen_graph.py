import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
from PIL import Image
import scipy
import cv2

import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
tf.disable_v2_behavior()

# The multiplication here gives space to generate direction with angle > pi/4
angle_mul = 12.0
output_dir = "OutputsInference/"

def postprocess_outputs(outputs):
    """Converts and normalize each map into a 3-channel image."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split original output into different maps
    partialOutputedNormals = outputs[:, :, :, 0:2] * angle_mul
    outputedDiffuse = outputs[:, :, :, 2:5]
    outputedRoughness = outputs[:, :, :, 5]
    outputedSpecular = outputs[:, :, :, 6:9]

    # [-1, 1] => [0, 1] and remove dummy dimension
    outputedDiffuse = (outputedDiffuse + 1) / 2
    outputedDiffuse = np.squeeze(outputedDiffuse)
    plt.imsave(os.path.join(output_dir, 'output_diffuse.png'), outputedDiffuse)

    # [-1, 1] => [0, 1] and remove dummy dimension
    outputedSpecular = (outputedSpecular + 1) / 2
    outputedSpecular = np.squeeze(outputedSpecular)
    plt.imsave(os.path.join(output_dir, 'output_specular.png'), outputedSpecular)

    # [-1, 1] => [0, 1] and remove dummy dimension. Also tiles 1-dim map to have RGB channels.
    outputedRoughness = np.expand_dims(outputedRoughness, axis=-1)
    outputedRoughness = np.tile(outputedRoughness, 3)
    outputedRoughness = (outputedRoughness + 1) / 2
    outputedRoughness = np.squeeze(outputedRoughness)
    plt.imsave(os.path.join(output_dir, 'output_roughness.png'), outputedRoughness)

    # Create new ones-array with same size that normals and concatenate, s.t. we have 3-channels.
    normal_shape = partialOutputedNormals.shape
    new_shape = [normal_shape[0], normal_shape[1], normal_shape[2], 1]
    tmpNormals = np.ones(new_shape, dtype=np.float32)
    normals = np.concatenate((partialOutputedNormals, tmpNormals), axis=-1)

    # From polar coordinates we normalize the normals.
    length = np.sqrt(np.sum(np.square(normals), axis=-1, keepdims=True))
    outputedNormals = normals / length

    # [-1, 1] => [0, 1] and remove dummy dimension
    outputedNormals = (outputedNormals + 1) / 2
    outputedNormals = np.squeeze(outputedNormals)
    plt.imsave(os.path.join(output_dir, 'output_normal.png'), outputedNormals)

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        # Load the graph in graph_def
        print("Loading Graph...")

        # We load the protobuf file from the disk and parse it to retrive the unserialized graph_def
        with gfile.FastGFile("OutputModels/output_graph_final_conv.pb", 'rb') as f:

            print("Loading Image...")
            input_image = mpimg.imread('Inputs/bark.png')
            input_image = input_image.astype(float)
            height, width, channels = input_image.shape
            print(height, width, channels)

            input_plot = plt.imshow(input_image)
            #plt.show()

            # Set FCN graph to the default graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)
            tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="",
                                op_dict=None, producer_op_list=None)

            # Print the name of operations in the session
            for op in graph.get_operations():
                print ("Operation Name :", op.name)  # Operation name
                print ("Tensor Stats :", str(op.values()))  # Tensor name

            # Set input/output nodes
            l_input = graph.get_tensor_by_name('preprocess_2/sub:0')  # Input Tensor
            l_output = graph.get_tensor_by_name('trainableModel/final_conv_last/Tanh:0')  # Output Tensor
            print("Shape of input : ", tf.shape(l_input))
            print("Shape of output : ", tf.shape(l_input))

            print("Shape of image before reshaping: ", input_image.shape)
            new_input_image = np.expand_dims(input_image, axis=(0, 1))
            print("Shape of image after reshaping: ", new_input_image.shape)

            outputs = sess.run(l_output, feed_dict={l_input: new_input_image})
            print(tf.shape(outputs))
            print(outputs.shape)

            postprocess_outputs(outputs)

