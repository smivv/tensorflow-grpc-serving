import argparse
import time
import yaml
import os

import grpc
import service.service_pb2 as service_pb
import service.service_pb2_grpc as service_grpc

from concurrent import futures

from classes.logger import Logger
from classes.classifier import Classifier

DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(DIR, 'settings.yml'), 'r') as f:
    SETTINGS = yaml.load(f)


def parse_args():
    parser = argparse.ArgumentParser()

    model = SETTINGS['default_model']

    parser.add_argument('--log_path', type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['log_path']), help='Path to the log file')

    parser.add_argument('--ip', required=False, type=str,
                        default=SETTINGS['server']['ip'], help='IP server will run on.')
    parser.add_argument('--port', required=False, type=int,
                        default=SETTINGS['server']['port'], help='Port server will run on.')

    parser.add_argument('--graph_path', type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['models'][model]['graph_path']),
                        help='Path to the graph file')

    parser.add_argument("--labels_path", type=str, required=False,
                        default=os.path.join(DIR, SETTINGS['models'][model]['labels_path']), help='Path to the labels file')

    parser.add_argument('--input_layer', type=str, default=SETTINGS['models'][model]['input_layer'],
                        help='Name of the input layer')

    parser.add_argument('--output_layer', type=str, default=SETTINGS['models'][model]['output_layer'],
                        help='Name of the output layer')

    parser.add_argument("--input_height", type=int, default=SETTINGS['image']['height'], help="Input image height")
    parser.add_argument("--input_width", type=int, default=SETTINGS['image']['width'], help="Input image width")
    parser.add_argument("--input_mean", type=int, default=SETTINGS['image']['mean'], help="Input image mean")
    parser.add_argument("--input_std", type=int, default=SETTINGS['image']['std'], help="Input image std")

    args = parser.parse_args()
    return args


FLAGS = parse_args()


class Server(service_grpc.RecognitionServicer):
    def __init__(self, config):
        super(Server, self).__init__()

        self.logger = Logger(config.log_path, 'server')

        self.logger.log("--- Initialization started ---")
        self.logger.info("--- Initialization started ---")

        start_time = time.time()

        self.classifier = Classifier(config.graph_path, config.labels_path,
                                     config.input_layer, config.output_layer,
                                     config.input_height, config.input_width,
                                     config.input_mean, config.input_std)

        self.logger.log("--- Total initialization took %s seconds ---" % (time.time() - start_time))
        self.logger.info("--- Total initialization took %s seconds ---" % (time.time() - start_time))

    def Recognize(self, request_iterator, context):
        try:
            # Extract image data from the message
            request = service_pb.Request()
            request.image.data = b''

            for r in request_iterator:
                request.image.format = r.image.format

                if r.image.data:
                    request.image.data += r.image.data

            self.logger.log("--- Message has been successfully received ---")
            self.logger.info("--- Message has been successfully received ---")

            if request.image.format not in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
                response = service_pb.Response()
                response.status.code = 0
                response.status.text = 'Not supported image format'

                self.logger.log("--- Not supported image format received ---")
                self.logger.info("--- Not supported image format received ---")

                return iter([response])

            filename = str(time.strftime("%Y-%m-%d %H:%M:%S"))
            filepath = os.path.join(DIR, 'images', filename + '.' + request.image.format)

            with open(filepath, 'wb') as file:
                file.write(request.image.data)
                file.close()

            self.logger.log("--- Recognition started ---")
            self.logger.info("--- Recognition started ---")

            start_time = time.time()

            output = self.classifier.recognize(filepath)

            self.logger.log("--- Recognition took %s seconds ---" % (time.time() - start_time))
            self.logger.info("--- Recognition took %s seconds ---" % (time.time() - start_time))

            # Create and send the response
            response = service_pb.Response()
            response.status.code = 1
            response.status.text = 'Success'

            for l, p in output:
                label = response.label.add()
                label.code = 1
                label.text = l
                label.probability = p

            self.logger.log("--- Recognition response sent ---")
            self.logger.info("--- Recognition response sent ---")

            os.remove(filepath)

            return iter([response])
        except Exception as e:
            self.logger.log("--- Error: %s ---" % str(e))
            self.logger.info("--- Error: %s ---" % str(e))
            response = service_pb.Response()
            response.status.code = 0
            response.status.text = 'Error: %s' % str(e)
            return iter([response])

# *****************************************************************************


def serve():
    logger = Logger(FLAGS.log_path, 'server')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    service = Server(FLAGS)

    service_grpc.add_RecognitionServicer_to_server(service, server)

    server.add_insecure_port('[::]:{}'.format(FLAGS.port))

    logger.log("--- Server has been started... ---")
    logger.info("--- Server has been started... ---")

    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':

    if not os.path.exists(os.path.join(DIR, FLAGS.graph_path)):
        raise FileNotFoundError('Graph file does not exist')

    if not os.path.exists(os.path.join(DIR, FLAGS.labels_path)):
        raise FileNotFoundError('Labels file does not exist')

    serve()
