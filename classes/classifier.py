import time
import numpy as np
import tensorflow as tf

from classes.logger import Logger


class Classifier:

    def __init__(self, graph_path, labels_path, input_layer, output_layer,
                 input_height=224, input_width=224, input_mean=0, input_std=255):

        self.logger = Logger()
        """ -------------------------------------- Original session -------------------------------------- """

        start_time = time.time()

        # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
        with tf.gfile.GFile(graph_path, "rb") as f:
            self.graph = tf.Graph()
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with self.graph.as_default():
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def)

        # Creating a session one time to reduce the time for serving a lot of images
        self.session = tf.Session(graph=self.graph)

        self.logger.log("--- Deep Neural Network session initialization took %s seconds ---" % (time.time() - start_time))
        self.logger.info("--- Deep Neural Network session initialization took %s seconds ---" % (time.time() - start_time))

        """ -------------------------------------- Input & output -------------------------------------- """
        # Access input and output nodes
        self.input_operation = self.graph.get_tensor_by_name('import/' + input_layer)
        self.output_operation = self.graph.get_tensor_by_name('import/' + output_layer)

        """ --------------------------------- Image preprocessing session ---------------------------------- """

        start_time = time.time()

        # Image processing graph
        self.image_graph = tf.Graph()
        with self.image_graph.as_default():
            self.image_path = tf.placeholder(tf.string)
            file_reader = tf.read_file(self.image_path, "file_reader")

            # Define image extension
            ext = tf.string_split([self.image_path], '.').values[1]

            def read_jpg(fr):
                return tf.image.decode_jpeg(fr, channels=3, name="jpeg_reader")

            def read_png(fr):
                return tf.image.decode_png(fr, channels=3, name="png_reader")

            def read_bmp(fr):
                return tf.image.decode_bmp(fr, name="bmp_reader")

            def read_gif(fr):
                return tf.image.decode_gif(fr, name="gif_reader")

            # Load image bytes
            image_reader = tf.case(
                {
                    tf.equal(ext, tf.constant('jpg', dtype=tf.string)):
                        lambda: read_jpg(file_reader),
                    tf.equal(ext, tf.constant('png', dtype=tf.string)):
                        lambda: read_png(file_reader),
                    tf.equal(ext, tf.constant('bmp', dtype=tf.string)):
                        lambda: read_bmp(file_reader),
                    tf.equal(ext, tf.constant('gif', dtype=tf.string)):
                        lambda: read_gif(file_reader)
                },
                default=lambda: read_jpg(file_reader),
                exclusive=True
            )

            float_caster = tf.cast(image_reader, tf.float32)
            dims_expander = tf.expand_dims(float_caster, 0)
            resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
            self.image_output = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

        self.image_sess = tf.Session(graph=self.image_graph)

        self.logger.log("--- Image preprocessing session initialization took %s seconds ---" % (time.time() - start_time))
        self.logger.info("--- Image preprocessing session initialization took %s seconds ---" % (time.time() - start_time))

        """ -------------------------------------- Labels loading -------------------------------------- """
        # Loading captions of labels
        self.labels = []
        self.load_labels(labels_path)

    # Launch image recognition session
    def recognize(self, image_path):

        start_time = time.time()

        data = self.read_tensor_from_image_file(image_path=image_path)

        self.logger.log("--- Image preprocessing took %s seconds ---" % (time.time() - start_time))
        self.logger.info("--- Image preprocessing took %s seconds ---" % (time.time() - start_time))

        start_time = time.time()

        probabilities = self.session.run(self.output_operation, feed_dict={
            self.input_operation: data
        })

        self.logger.log("--- DNN inference took %s seconds ---" % (time.time() - start_time))
        self.logger.info("--- DNN inference took %s seconds ---" % (time.time() - start_time))

        probabilities = np.squeeze(probabilities)

        return [(self.labels[i], prob) for i, prob in enumerate(probabilities)]

    # Launch image preprocessing session
    def read_tensor_from_image_file(self, image_path):
        return self.image_sess.run(self.image_output, feed_dict={
            self.image_path: image_path
        })

    # Load labels from file
    def load_labels(self, label_path):
        for l in open(label_path, 'r').readlines():
            self.labels.append(l.split(':')[1][:-1])
        return self.labels
