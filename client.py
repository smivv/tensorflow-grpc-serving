import argparse
import time
import math
import yaml
import os

import grpc
import service.service_pb2 as service_pb
import service.service_pb2_grpc as service_grpc

from classes.logger import Logger

DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(DIR, 'settings.yml'), 'r') as f:
    SETTINGS = yaml.load(f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ip', required=False, type=str,
                        default=SETTINGS['server']['ip'], help='IP server will run on.')
    parser.add_argument('--port', required=False, type=str,
                        default=SETTINGS['server']['port'], help='Port server will run on.')

    parser.add_argument('--image_path', type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['image_path']), help='Path to the image file')

    parser.add_argument('--log_path', type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['log_path']), help='Path to the log file')

    args = parser.parse_args()
    return args


FLAGS = parse_args()


class Client(object):
    TOP_K = 3

    def __init__(self, config):

        if not os.path.exists(FLAGS.log_path):
            os.mkdir(FLAGS.log_path)
        self.logger = Logger(config.log_path, 'client')

        self.stub = service_grpc.RecognitionStub(
            grpc.insecure_channel('{}:{}'.format(FLAGS.ip, FLAGS.port))
        )

    def RecognizeTest(self, filepath):
        self.logger.log('--- Performing recognition ---')
        self.logger.info('--- Performing recognition ---')

        request = service_pb.Request()

        # Extract data from image file
        with open(filepath, 'rb') as file:
            request.image.data = file.read()

        # Define file extension
        request.image.format = os.path.basename(filepath).split('.')[-1]

        start_time = time.time()

        # Perform request
        responses = self.stub.Recognize(iter([request]))

        results = None
        for response in responses:
            if response.status.code == 0:
                print('Result = {} '.format(response.status.text))
                return False

            if len(response.label) == 0:
                return False

            results = sorted(response.label, key=lambda l: -l.probability)[:self.TOP_K]

        end_time = (time.time() - start_time)
        self.logger.log("--- Recognition took %s seconds ---" % end_time)
        self.logger.info("--- Recognition took %s seconds ---" % end_time)

        print('Results:')
        for result in results:
            print("'{}' with probability {}%.".format(result.text, math.floor(result.probability * 100)))
        return True

# *****************************************************************************


def serve():
    client = Client(FLAGS)

    client.RecognizeTest(FLAGS.image_path)


if __name__ == '__main__':

    if not os.path.exists(FLAGS.image_path):
        raise FileNotFoundError('Image file does not exist')

    serve()
