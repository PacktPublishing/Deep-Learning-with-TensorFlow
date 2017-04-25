import tensorflow as tf, sys

# You will be sending the image to be classified as a parameter
provided_image_path = sys.argv[1]

# then we will read the image data
provided_image_data = tf.gfile.FastGFile(provided_image_path, 'rb').read()

# Loads label file
label_lines = [line.rstrip() for line 
             in tf.gfile.GFile("tensorflow_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("tensorflow_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # pass the provided_image_data as input to the graph
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    netowrk_predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': provided_image_data})
    
    # Sort the result by confidence to show the flower labels accordingly
    top_predictions = netowrk_predictions[0].argsort()[-len(netowrk_predictions[0]):][::-1]
    
    for prediction in top_predictions:
        flower_type = label_lines[prediction]
        score = netowrk_predictions[0][prediction]
        print('%s (score = %.5f)' % (flower_type, score))
