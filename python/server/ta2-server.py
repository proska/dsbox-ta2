#!/usr/bin/env python

import os
import sys
import os.path

# Setup Paths
PARENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENTDIR)

from dsbox_dev_setup import path_setup
path_setup()

import time
import grpc
import numpy
import argparse
from concurrent import futures

import core_pb2_grpc
from dsbox.server.ta2_servicer import TA2Servicer

import multiprocessing
from multiprocessing import Pool

numpy.set_printoptions(threshold=numpy.nan)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEFAULT_LIB_DIRECTORY = PARENTDIR + os.sep + "library"

PORT = 45042

def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--library", dest="library",
        help="Primitives library directory. [default: %(default)s]",
        default=DEFAULT_LIB_DIRECTORY)
    args = parser.parse_args()

    servicer = TA2Servicer(args.library)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    library = args.library

    core_pb2_grpc.add_CoreServicer_to_server(servicer, server)

    server.add_insecure_port('[::]:' + str(PORT))
    server.start()

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
