import argparse
import time
import yaml
import os

from classes.classifier import Classifier
from classes.logger import Logger

DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(DIR, 'settings.yml'), 'r') as f:
    SETTINGS = yaml.load(f)


def parse_args():
    parser = argparse.ArgumentParser()

    default_model = SETTINGS['default_model']

    parser.add_argument('--log_path', type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['log_path']), help='Path to the log file')

    parser.add_argument('--image_path', type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['image_path']), help='Path to the image file')

    parser.add_argument('--graph_path', type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['models'][default_model]['graph_path']),
                        help='Path to the graph file')

    parser.add_argument('--input_layer', type=str,
                        default=SETTINGS['models'][default_model]['input_layer'],
                        help='Name of the input layer')
    parser.add_argument('--output_layer', type=str,
                        default=SETTINGS['models'][default_model]['output_layer'],
                        help='Name of the output layer')

    parser.add_argument("--labels_path", type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['models'][default_model]['labels_path']),
                        help='Path to the labels file')

    parser.add_argument("--input_height", type=int, default=SETTINGS['image']['height'], help="Input image height")
    parser.add_argument("--input_width", type=int, default=SETTINGS['image']['width'], help="Input image width")
    parser.add_argument("--input_mean", type=int, default=SETTINGS['image']['mean'], help="Input image mean")
    parser.add_argument("--input_std", type=int, default=SETTINGS['image']['std'], help="Input image std")

    args = parser.parse_args()
    return args


FLAGS = parse_args()

if __name__ == '__main__':

    if not os.path.exists(FLAGS.graph_path):
        raise FileNotFoundError('Graph file does not exist')

    if not os.path.exists(FLAGS.labels_path):
        raise FileNotFoundError('Labels file does not exist')

    if not os.path.exists(FLAGS.log_path):
        os.mkdir(FLAGS.log_path)
    logger = Logger(FLAGS.log_path, 'inference')

    start_time = time.time()
    classifier = Classifier(FLAGS.graph_path, FLAGS.labels_path,
                            FLAGS.input_layer, FLAGS.output_layer,
                            FLAGS.input_height, FLAGS.input_width,
                            FLAGS.input_mean, FLAGS.input_std)

    logger.log("--- Initialization took %s seconds ---" % (time.time() - start_time))
    logger.info("--- Initialization took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    result = classifier.recognize(FLAGS.image_path)
    logger.log("--- %s seconds ---" % (time.time() - start_time))
    logger.info("--- %s seconds ---" % (time.time() - start_time))
