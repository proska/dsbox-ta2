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

PORT = 45042

def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output', help='Output directory', default='')
    parser.add_argument(
        '--debug-volume-map', action='append',
        help="Map config directories, e.g. --debug-volume-map /host/dir/output:/output --debug-volume-map /host/dir/input:/input",
        default=[])
    parser.add_argument(
        '--load-pipeline', help='Load using fitted pipeline ID')
    args = parser.parse_args()

    print(args)

    dir_mapping = {}
    for entry in args.debug_volume_map:
        host_dir, container_dir = entry.split(':')
        dir_mapping[host_dir] = container_dir
        print('volume: {} to {}'.format(host_dir, container_dir))

    output_dir = args.output
    if output_dir=='':
        if 'D3MOUTPUTDIR' in os.environ:
            output_dir = os.path.abspath(os.environ['D3MOUTPUTDIR'])
        else:
            output_dir = '/output'

    servicer = TA2Servicer(
        output_dir=output_dir,
        directory_mapping=dir_mapping,
        fitted_pipeline_id=args.load_pipeline)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

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
