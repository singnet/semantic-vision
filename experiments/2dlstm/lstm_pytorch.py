import queue
from enum import IntEnum
import numpy
import torch
import torch.nn as nn


HEIGHT = -2
WIDTH = -1


class Direction(IntEnum):
   RIGHT_DOWN=1
   RIGHT_TOP=2
   LEFT_DOWN=3
   LEFT_TOP=4
   DOWN_RIGHT=5
   TOP_RIGHT=6
   DOWN_LEFT=7
   TOP_LEFT=8


# direction over (x, y) axis
# -1 means reversed order
# 1  means default order
# from left to right
# from top to bottom
MULT_DIRECTIONS = { Direction.RIGHT_DOWN:(1, 1),
                    Direction.RIGHT_TOP: (1, -1),
                    Direction.LEFT_DOWN:  (-1, 1),
                    Direction.LEFT_TOP: (-1, -1),
                    Direction.DOWN_RIGHT: (1, 1),
                    Direction.TOP_RIGHT: (1, -1),
                    Direction.DOWN_LEFT: (-1, 1),
                    Direction.TOP_LEFT: (-1, -1)}


class Axis(IntEnum):
    X = 0
    Y = 1


class LstmIterator:
    def __init__(self, tensor, size_x, size_y, direction=Direction.RIGHT_DOWN):
        """
        (30, 3, 128, 128)
        The return type must be duplicated in the docstring to comply
        with the NumPy docstring style.

        Parameters
        ----------
        tensor
            array with 4 dimentions: (#batch, #channel, #HEIGHT, #WIDTH)
        size_x
            WIDTH in number of pixels
        size_y
            HEIGHT in number of pixels
        direction
            lstm_pytorch.Direction's field, determins order of direction
            for the iteration over image
        """
        self.size_x = size_x
        self.size_y = size_y
        self.tensor = self.zero_pad(tensor, size_x, WIDTH)
        self.tensor = self.zero_pad(self.tensor, size_y, HEIGHT)
        assert(self.tensor.shape[WIDTH] % size_x == 0)
        assert(self.tensor.shape[HEIGHT] % size_y == 0)
        self.steps_x = self.tensor.shape[WIDTH] // size_x
        self.steps_y = self.tensor.shape[HEIGHT] // size_y
        self.max_steps = self.steps_x * self.steps_y
        self.out_shape = (tensor.shape[0], tensor.shape[1], self.steps_y, self.steps_x)
        self.direction = direction

    def _get_iterator(self, axis, direction):
        if axis == Axis.X:
            steps = self.steps_x
        else:
            steps = self.steps_y
        if 0 < direction:
            return range(steps)
        return reversed(range(steps))

    def __iter__(self):
        size_x = self.size_x
        size_y = self.size_y
        if self.direction <= Direction.LEFT_TOP:
            first_axis = Axis.Y
            second_axis = Axis.X
        else:
            first_axis = Axis.X
            second_axis = Axis.Y

        for i in self._get_iterator(first_axis, MULT_DIRECTIONS[self.direction][first_axis]):
            for j in self._get_iterator(second_axis, MULT_DIRECTIONS[self.direction][second_axis]):
                if first_axis == Axis.X:
                    pos = (j, i)
                else:
                    pos = (i, j)
                yield self.__get_item(*pos)

    def __get_item(self, i, j):
        size_x = self.size_x
        size_y = self.size_y
        result = self.tensor[:, :,
                          i * size_x: i * size_x + size_x,
                          j * size_y: j * size_y + size_y]
        return result.reshape((1, result.shape[0], numpy.prod(result.shape[1:])))

    @staticmethod
    def zero_pad_size(dim, step):
        return (step - dim % step) * bool(dim % step)

    def zero_pad(self, tensor, step, axis):
        pad = self.zero_pad_size(tensor.shape[axis], step)
        if pad:
            pad_shape = list(tensor.shape)
            pad_shape[axis] = pad
            return torch.cat((tensor, torch.zeros(pad_shape, dtype=tensor.dtype)), axis)
        return tensor

    @property
    def look_back_idx(self):
        if self.direction <= Direction.LEFT_TOP:
            return self.steps_x
        return self.steps_y

    @property
    def batch_size(self):
        return self.tensor.shape[0]


class Lstm2D(nn.Module):

    def __init__(self, input_size, num_cells, neib_shape):
        super(Lstm2D, self).__init__()
        self._input_size = input_size
        self._num_cells = num_cells
        self.lstm = nn.LSTM(input_size + num_cells, num_cells)
        self.neib_shape = neib_shape
        self._queue = None
        self._outs = None
        self._out_shape = None

    def forward(self, batch):
        iterator = LstmIterator(batch, *self.neib_shape)
        self._queue = queue.Queue()
        self._outs = []
        out_shape = list(iterator.out_shape)
        out_shape[1] = self._num_cells
        self._out_shape = out_shape
        out_prev = torch.zeros((1, iterator.batch_size, self._num_cells))
        c = torch.zeros(( 1, iterator.batch_size, self._num_cells))
        hidden = [out_prev, c]
        # loop over minibatch
        for i, item in enumerate(iterator):
            if i < iterator.look_back_idx:
                prev_2 = torch.zeros(1, iterator.batch_size, self._num_cells, dtype=torch.float)
            else:
                prev_2 = self._queue.get()
            inp = torch.cat([item, prev_2], dim=2)
            out, hidden = self.lstm(inp, hidden)
            self._outs.append(out)
            self._queue.put(out)
        return self.to_image(self._outs)

    def to_image(self, out):
        return torch.cat(out).permute((1,0,2)).reshape(self._out_shape)

