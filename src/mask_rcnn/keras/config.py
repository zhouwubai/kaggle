from pprint import pprint
import numpy as np


class Config(object):

    name = None
    gpu_count = 1
    images_per_gpu = 2

    # train
    steps_per_epoch = 1000
    validation_steps = 50

    backbone = "resnet101"
    backbone_strides = [4, 8, 16, 32, 64]

    num_classes = 1

    # rpn
    rpn_anchor_scales = [32, 64, 128, 256, 512]
    rpn_anchor_ratios = [0.5, 1, 2]
    rpn_anchor_stride = 1
    rpn_nms_threshold = 0.7
    rpn_train_anchor_per_image = 256
    post_nms_rois_training = 2000
    pos_nms_rois_inference = 1000

    use_mini_mask = True
    mini_mask_shape = (56, 56)

    # image scale
    image_resize_mode = "square"
    image_min_dim = 800
    image_max_dim = 1024
    image_min_scale = 0
    mean_pixel = np.array([123.7, 116.8, 103.9])

    # roi
    train_rois_per_image = 200
    roi_positive_ratio = 0.33
    pool_size = 7
    mask_pool_size = 14
    mask_shape = [28, 28]
    max_gt_instances = 100

    # predict
    rpn_bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
    bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])

    detection_min_confidence = 0.7
    detection_nms_threshold = 0.3

    # training
    learning_rate = 0.001
    learning_momentum = 0.9
    weight_decay = 0.0001

    loss_weights = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    use_rpn_rois = True
    train_bn = False
    gradient_clip_norm = 5.0

    batch_size = images_per_gpu * gpu_count

    img_h = image_max_dim

    image_shape = np.array([img_h, img_h, 3])

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
