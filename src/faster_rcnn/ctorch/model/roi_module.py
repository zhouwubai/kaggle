from collections import namedtuple
from string import Template

import cupy
import torch
import cupy as cp
import torch as t
from torch.autograd import Function

from faster_rcnn.ctorch.model.utils.roi_cupy import (
    kernel_backward, kernel_forward
)

Stream = namedtuple('Stream', ['ptr'])


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    """
    return:
        `cupy.cuda.Function` def __call__(self, tuple grid, tuple block,
                                          args, size_t shared_mem=0,
                                          stream=None)
        block=(N, 1, 1) means its a 1D indexing
    """
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


class RoI(Function):
    """
    NOTE：only CUDA-compatible
    Backend function for ROIPooling layer

    This outputs fixed size feature map for each roi pooled from the
    feature map of whole image based on roi location

    Args:
        outh (int): the height of output feature map
        outw (int): the weight of output feature map
        spatial_scale: scale of roi is resized
    """

    def __init__(self, outh, outw, spatial_scale):
        self.forward_fn = load_kernel('roi_forward', kernel_forward)
        self.backward_fn = load_kernel('roi_backward', kernel_backward)
        self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale

    def forward(self, x, rois):
        """
        NOTE: MAKE SURE input is contiguous so it can be correctly addressed
        in C code

        Args:
            x (Variable): 4D image variable (feature map from vgg16)
            rois (Variable): rois with indices for image in the batch
                shape (R, 5) => (ind, y_min, x_min, y_max, x_max)
        """
        x = x.contiguous()
        rois = rois.contiguous()
        self.in_size = B, C, H, W = x.size()
        self.N = N = rois.size(0)
        output = t.zeros(N, C, self.outh, self.outw).cuda()
        self.argmax_data = t.zeros(N, C, self.outh, self.outw).int().cuda()
        self.rois = rois
        # data_ptr(), the address of first element in the array
        args = [x.data_ptr(), rois.data_ptr(),
                output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.spatial_scale, C, H, W,
                self.outh, self.outw,
                output.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.forward_fn(args=args,
                        block=(CUDA_NUM_THREADS, 1, 1),
                        grid=(GET_BLOCKS(output.numel()), 1, 1),
                        stream=stream)
        return output

    def backward(self, grad_output):
        # NOTE: IMPORTANT CONTIGUOUS
        # TODO: input
        grad_output = grad_output.contiguous()
        B, C, H, W = self.in_size
        grad_input = t.zeros(self.in_size).cuda()
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.rois.data_ptr(),
                grad_input.data_ptr(),
                self.N, self.spatial_scale, C, H, W,
                self.outh, self.outw, grad_input.numel()]
        self.backward_fn(args=args,
                         block=(CUDA_NUM_THREADS, 1, 1),
                         grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
                         stream=stream
                         )
        return grad_input, None


class RoIPooling2D(t.nn.Module):
    """ROIPooling layer
    This class is used as the ROIPooling layer for faster R-CNN
    This outputs fixed size feature map for each roi pooled from the
    feature map of whole image based on roi location

    Args:
        outh (int): the height of output feature map
        outw (int): the weight of output feature map
        spatial_scale: scale of roi is resized
    """

    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.RoI = RoI(outh, outw, spatial_scale)

    def forward(self, x, rois):
        """Forward the chain

        We assume that there are :math: `N` batches.

        Args:
            x (Variable): 4D image variable (feature map from vgg16)
            rois (Variable): rois with indices for image in the batch
                shape (R, 5) => (ind, y_min, x_min, y_max, x_max)
        """
        return self.RoI(x, rois)


def test_roi_module():
    # fake data###
    B, N, C, H, W, PH, PW = 2, 8, 4, 32, 32, 7, 7

    bottom_data = t.randn(B, C, H, W).cuda()
    bottom_rois = t.randn(N, 5)
    bottom_rois[:int(N / 2), 0] = 0
    bottom_rois[int(N / 2):, 0] = 1
    bottom_rois[:, 1:] = (t.rand(N, 4) * 100).float()
    bottom_rois = bottom_rois.cuda()
    spatial_scale = 1. / 16
    outh, outw = PH, PW

    # pytorch version
    module = RoIPooling2D(outh, outw, spatial_scale)
    x = t.autograd.Variable(bottom_data, requires_grad=True)
    rois = t.autograd.Variable(bottom_rois)
    output = module(x, rois)
    output.sum().backward()

    def t2c(variable):
        npa = variable.data.cpu().numpy()
        return cp.array(npa)

    def test_eq(variable, array, info):
        cc = cp.asnumpy(array)
        neq = (cc != variable.data.cpu().numpy())
        assert neq.sum() == 0, 'test failed: %s' % info

    # chainer version,if you're going to run this
    # pip install chainer
    import chainer.functions as F
    from chainer import Variable
    x_cn = Variable(t2c(x))

    o_cn = F.roi_pooling_2d(x_cn, t2c(rois), outh, outw, spatial_scale)
    test_eq(output, o_cn.array, 'forward')
    F.sum(o_cn).backward()
    test_eq(x.grad, x_cn.grad, 'backward')
    print('test pass')