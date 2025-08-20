import numpy as np
import cv2
from is_msgs.common_pb2 import Tensor, DataType
from is_msgs.image_pb2 import Image
import sys
import math

def image_to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image


def np_to_tensor(array):
    tensor = Tensor()
    if len(array.shape) == 0:
        return tensor
    if len(array.shape) > 2:
        raise Exception('Implemented only for one or two dimensional np.ndarray')
    dims_name = ['rows', 'cols']
    for size, name in zip(array.shape, dims_name):
        dim = tensor.shape.dims.add()
        dim.size = size
        dim.name = name
    dtype = array.dtype
    if dtype in ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']:
        tensor.type = DataType.Value('INT32_TYPE')
        tensor.ints32.extend(array.ravel().tolist())
    elif dtype in ['int64', 'uint64']:
        tensor.type = DataType.Value('INT64_TYPE')
        tensor.ints64.extend(array.ravel().tolist())
    elif dtype in ['float16', 'float32']:
        tensor.type = DataType.Value('FLOAT_TYPE')
        tensor.floats.extend(array.ravel().tolist())
    elif dtype in ['float64']:
        tensor.type = DataType.Value('DOUBLE_TYPE')
        tensor.doubles.extend(array.ravel().tolist())
    else:
        pass
    return tensor


def tensor_to_np(tensor):
    if len(tensor.shape.dims) != 2 or tensor.shape.dims[0].name != 'rows':
        return np.array([])
    shape = (tensor.shape.dims[0].size, tensor.shape.dims[1].size)
    if tensor.type == DataType.Value('INT32_TYPE'):
        return np.asarray(tensor.ints32, dtype=np.int32).reshape(shape)
    if tensor.type == DataType.Value('INT64_TYPE'):
        return np.asarray(tensor.ints64, dtype=np.int64).reshape(shape)
    if tensor.type == DataType.Value('FLOAT_TYPE'):
        return np.asarray(tensor.floats, dtype=np.float32).reshape(shape)
    if tensor.type == DataType.Value('DOUBLE_TYPE'):
        return np.asarray(tensor.doubles, dtype=np.float64).reshape(shape)
    return np.array([])

def wait_for_reply(channel,request_id,timeout=0.5,max_tries = 10):
        n_try = 0
        final_reply =None
        while n_try<max_tries:
            if timeout is None:
                reply = channel.consume()
            else:
                reply = channel.consume(timeout=timeout)
            if reply.correlation_id == request_id:
                final_reply = reply
                n_try = max_tries+1        
            else:
                n_try+=1
        return final_reply   

def _yaw_to_quat(yaw: float):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (0.0, 0.0, sy, cy)  # roll=pitch=0
     
from is_wire.core import Channel
from is_wire.core.wire.conversion import WireV1


class StreamChannel(Channel):
    def consume(self, return_dropped=False):
        def clean_and_consume(timeout=None):
            self.amqp_message = None
            while self.amqp_message is None:
                self.connection.drain_events(timeout=timeout)
            return self.amqp_message

        _amqp_message = clean_and_consume()
        dropped = 0
        while True:
            try:
                # will raise an exceptin when no message remained
                _amqp_message = clean_and_consume(timeout=0.0)
                dropped += 1
            except:
                # returns last message
                msg = WireV1.from_amqp_message(_amqp_message)
                return (msg, dropped) if return_dropped else msg

