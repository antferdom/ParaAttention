import os
import sys
import time
import uuid
import signal
import socket
import inspect
import torch
import torch.multiprocessing as mp
from flask import Flask, jsonify, request

from .inference_worker import inference_worker

HTTP_STATUS = {
    'OK': 200,
    'ACCEPTED': 202,
    'NOT_FOUND': 404,
    'BAD_REQUEST': 400,
    'UNAUTHORIZED': 401,
    'FORBIDDEN': 403,
    'INTERNAL_SERVER_ERROR': 500,
    'SERVICE_UNAVAILABLE': 503,
}


class AsyncGenerationServer:
    def __init__(self, model, host : str, port: int, num_devices: int):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.torch_distributed_port = 29500
        self.world_size = torch.cuda.device_count() if num_devices is None else num_devices
        print(f"server running on port {self.port} with {self.world_size} devices")

        # register signal handler for SIGTERM/SIGINT
        signal.signal(signal.SIGTERM, self._create_signal_handler())
        signal.signal(signal.SIGINT, self._create_signal_handler())

        if not self._is_port_available(self.port):
            raise ValueError(f"port {self.port} is already in use")

        # look for a free port for torch distributed. This is a less than ideal hack
        # its mostly required for testing, sometimes running inference on a model will 
        # result in worker processes being uninterruptable, which prevents the signal handler
        # from firing when the server terminates the worker process at the end of the class
        # lifetime. This prevents the subprocess from being able to call dist.destroy_process_group()
        # which is required if we want to resuse the port. This shouldn't be an issue in production, because
        # the server wont be terminated/restarted, but it is a problem for testing.
        while not self._is_port_available(self.torch_distributed_port):
            self.torch_distributed_port += 1
        
        # for sending requests to workers
        self.in_queue = [mp.Queue() for _ in range(self.world_size)]
        
        # for workers communicating status and errors back to host
        self.worker_error_msg_queue = [mp.Queue() for _ in range(self.world_size)]
        
        # worker_status[i] == 0 means worker with rank i is good
        # any worker_status[i] != 0 means error occured
        self.worker_status = [mp.Value('i', 0) for _ in range(self.world_size)]

        # an event that is used to communicate to workers that a request has been cancelled
        # it is set by the server, and waited on by a dedicated thread within the worker processes
        self.cancel_event = mp.Event()
        
        # the model to be used for inference with setup() and predict() methods implemented by user
        # we will throw an exception here is the model is not valid 
        self.model = model
        self._validate_model(self.model)

        # get arguments names of all non-optional arguments of self.model.predict
        self.predict_args = inspect.signature(self.model.predict).parameters.keys()

        # an unsigned integer counter to track how many workers are available
        # to handle an incoming request
        self.ready_workers = mp.Value('i', 0)

        # bookkeeping for current request status
        self.current_request_id = None
        
        # set mp start method to spawn, required if subprocesses use CUDA
        self._init_mp()
        
        # spawn worker processes
        self.worker_processes = []
        for i in range(self.world_size):

            worker_args = (
                self.in_queue[i],
                self.worker_error_msg_queue[i],
                self.worker_status[i],
                self.cancel_event,
                i,
                self.world_size,
                self.model,
                self.ready_workers,
                self.torch_distributed_port,
            )
            
            worker_process = mp.Process(target=inference_worker, args=worker_args)
            self.worker_processes.append(worker_process)
            worker_process.start()
        
        # wait for workers to initialize
        # TODO test what happens if a worker throws an exception during setup()
        while True:
            time.sleep(1)
            print(f"server {os.getpid()} waiting for workers to initialize ...")
            
            # check if any workers have encountered an error
            # if an error occurs in a worker process during setup, we will rethrow it here
            # and the server will exit
            self._are_workers_good()
            
            # check if all workers are ready
            if self.ready_workers.value == self.world_size:
                print(f"server {os.getpid()} workers initialized!")
                break
        
        # register routes
        self.app.route('/generate', methods=['POST'])(self.generate)
        self.app.route('/request_status/<request_id>', methods=['POST'])(self.request_status)
        self.app.route('/cancel', methods=['POST'])(self.cancel)
        self.app.route('/server_status', methods=['GET'])(self.server_status)

    # destructor
    def __del__(self):
        print(f"server {os.getpid()} destructor called")
        self._cleanup()

    def _create_signal_handler(self):
        """
        create a signal handler for the server
        which is just _cleanup() + exit(0)
        """
        def handler(signum, frame):
            print(f"server {os.getpid()} signal handler called")
            self._cleanup()
            sys.exit(0)
        return handler
    
    def _cleanup(self):
        """
        cleanup function for the server, send SIGTERM, and join all worker processes
        should be idempotent, doesn't matter if it is called multiple times
        """

        if hasattr(self, 'worker_processes') and self.worker_processes:
            print(f"server {os.getpid()} _cleanup called")
            for worker in self.worker_processes:
                worker.terminate()

            for worker in self.worker_processes:
                worker.join(timeout=1)
                if worker.is_alive():
                    print(f"WARNING: server {os.getpid()} worker {worker.pid} unresponsive to SIGTERM, sending SIGKILL")
                    os.kill(worker.pid, signal.SIGKILL)
            print(f"server {os.getpid()} cleaned up")
        self.worker_processes = None

    def _are_workers_good(self):
        """
        check whether all workers are alive and well, if not will raise an exception
        if so will return True
        """
        print(f"server {os.getpid()} _are_workers_good called")
        for i in range(self.world_size):
            if not self.worker_processes[i].is_alive():
                raise ValueError(f"worker {i} is not alive")
            if self.worker_status[i].value != 0:
                error_msg = self.worker_error_msg_queue[i].get_nowait()
                raise ValueError(f"server {os.getpid()} worker {i} has encountered an error {error_msg}")
        print(f"server {os.getpid()} worker status is good")
        return True


    def _is_port_available(self, port, host='localhost'):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            return True
        except socket.error:
            return False
        finally:
            sock.close()

    def _validate_request(self, request_kwargs):
        """
        all keys in request_kwargs must be arguments to predict
        a output_path must be provided
        """
        return all(k in self.predict_args for k in request_kwargs.keys()) and 'output_path' in request_kwargs

    def _validate_model(self, model):
        """
        model must have methods setup and predict
        model.predict method must have an argument 'output_path'
        """
        if not hasattr(model, 'setup'):
            raise ValueError("model must have a setup method")
        if not hasattr(model, 'predict'):
            raise ValueError("model must have a predict method")
        if 'output_path' not in inspect.signature(model.predict).parameters:
            raise ValueError("model.predict method must have an argument 'output_path'")

    def _init_mp(self):
        """
        set mp start method to spawn, required if subprocesses use CUDA
        """
        current_method = mp.get_start_method(allow_none=True)
        if current_method != "spawn":
            print(f"{os.getpid()} setting start method to spawn")
            mp.set_start_method('spawn', force=True)
    
    def generate(self):
        """
        Handle generation requests.
        Returns a request ID that can be used to check status.
        """
        try:
            self._are_workers_good()

            kwargs = request.get_json()
            if not self._validate_request(kwargs):
                # bad request, request contains invalid arguments
                return jsonify({'error': f'request contains invalid arguments'}), HTTP_STATUS['BAD_REQUEST']
            # print(f"server recieved request {kwargs}")

            # if workers are not all ready, sleep for 100ms and check again
            # if we just finished serving the previous request, it may take an additional instant
            # or two for the workers to all be ready again
            seconds_to_wait = 0
            while self.ready_workers.value != self.world_size and seconds_to_wait < 5:
                time.sleep(0.1)
                seconds_to_wait += 0.1
            
            # if workers are not all ready after waiting for 5 seconds, one of them probably hung
            if self.ready_workers.value != self.world_size:
                raise ValueError(f"server {os.getpid()}: only {self.ready_workers.value} workers ready, this is probably a bug")
                    
            # request ids are used for tracking of in flight requests
            # bookkeep the request status and output_path
            self.current_request_id = str(uuid.uuid4())
            self.current_output_path = kwargs['output_path']

            # ensure all in_queues are empty
            # since this server only handles one request at a time
            # TODO if we are going to allow the server to queue requests get rid of this check
            # right now server is handling only one request at a time
            for ix in range(self.world_size):
                if not self.in_queue[ix].empty():
                    raise ValueError(f"in_queue {ix} is not empty")
            
            # if the file that is going to be written to already exists, remove it
            # this file existing on disk is a criteria server uses to determine if a request is complete
            if os.path.exists(self.current_output_path):
                os.remove(self.current_output_path)

            # send the request to all workers and check that all workers are good
            for i in range(self.world_size):
                self.in_queue[i].put((self.current_request_id, kwargs))

            print(f"server {os.getpid()} request {self.current_request_id} accepted")
            return jsonify({'request_id': self.current_request_id}), HTTP_STATUS['ACCEPTED']
        except Exception as e:
            return jsonify({'internal server error': str(e)}), HTTP_STATUS['INTERNAL_SERVER_ERROR']

    
    def request_status(self, request_id):
        """
        Check the status of a generation request.
        """
        try:
            self._are_workers_good()

            if self.current_request_id is None:
                return jsonify({'status': f'no request in flight'}), HTTP_STATUS['NOT_FOUND']
            
            elif self.current_request_id != request_id:
                return jsonify({'status': f'request {request_id} not found, current request id is {self.current_request_id}'}), HTTP_STATUS['NOT_FOUND']
            
            elif self.current_request_id == request_id:
                # at this point we know that the request is in flight
                # there are three cases:
                
                if os.path.exists(self.current_output_path) and self.ready_workers.value == self.world_size:
                    # case 1: worker statuses are all OK and the file exists on disk, request is complete
                    response = jsonify({'status': f'request {request_id} is completed', 'output_path': f"{self.current_output_path}"})
                    
                    # reset the request id and output_path, this request is now considered complete by the server
                    self.current_request_id = None
                    self.current_output_path = None
                    return response, HTTP_STATUS['OK']
                else:
                    # case 2: worker statuses are all OK and the file does not exist on disk, request is still in flight
                    return jsonify({'status': f'request {request_id} is in flight'}), HTTP_STATUS['ACCEPTED']
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    
    def server_status(self):
        """
        check whether the server is here or not
        """
        try:
            self._are_workers_good()
            return jsonify({'status': 'server is running'}), HTTP_STATUS['OK']
        except Exception as e:
            return jsonify({'error': str(e)}), HTTP_STATUS['INTERNAL_SERVER_ERROR']


    def cancel(self):
        """
        Cancel an ongoing generation request.
        """
        try:
            if self.current_request_id is None:
                return jsonify({'error': f'there is currently no request in flight to cancel'}), HTTP_STATUS['NOT_FOUND']
            else:
                request_id = self.current_request_id
            
            self.cancel_event.set()
            while self.ready_workers.value < self.world_size:
                time.sleep(1)
                self._are_workers_good()
            self.cancel_event.clear()
            self.current_request_id = None
            self.current_output_path = None
            return jsonify({'status': f'request {request_id} cancelled'}), HTTP_STATUS['OK']
        except Exception as e:
            return jsonify({'error': str(e)}), HTTP_STATUS['INTERNAL_SERVER_ERROR']

    def run(self):
        """
        Start the Flask server.
        """
        self.app.run(host=self.host, port=self.port)