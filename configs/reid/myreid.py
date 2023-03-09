TRAIN_REID = True
_base_ = [
    '../_base_/datasets/mot_challenge_reid.py', '../_base_/default_runtime.py'
]
model = dict(reid=dict(  # 重识别模型的配置
        type='BaseReID',  # 重识别模型的名称
        backbone=dict(  # 重识别模型的主干网络配置
            type='ResNet', # 详细请查看 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L288 了解更多的主干网络
            depth=50,  # 主干网络的深度，对于 ResNet 以及 ResNext 网络，通常使用50或者101深度
            num_stages=4,  # 主干网络中阶段的数目
            out_indices=(3, ),  # 每个阶段产生的输出特征图的索引
            style='pytorch',
            # init_cfg=dict(
            #     type='Pretrained',
            #     checkpoint=  # noqa: E251
            #     'torchvision://resnet50'  # noqa: E501
            # )
        ),  # 主干网络的形式，'pytorch' 表示步长为2的网络层在3x3的卷积中，'caffe' 表示步长为2的网络层在1x1卷积中。
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),  # 重识别模型的颈部,通常是全局池化层。
        head=dict(  # 重识别模型的头部
            type='LinearReIDHead',  # 分类模型头部的名称
            num_fcs=1,  # 模型头部的全连接层数目
            in_channels=2048,  # 输入通道的数目
            fc_channels=1024,  # 全连接层通道数目
            out_channels=128,  # 输出通道数目
            norm_cfg=dict(type='BN1d'),  # 规一化模块的配置
            act_cfg=dict(type='ReLU'), # 激活函数模块的配置
            loss=dict(type='TripletLoss', loss_weight=1.0),
            num_classes=1),
        ))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.01,
    step=[3])
# data
data_root = 'data/'
data = dict(
    train=dict(data_prefix=data_root + 'MOT17/train', ann_file=data_root + 'cocodata/train_cocoformat.json'),
    val=dict(data_prefix=data_root + 'MOT17/train', ann_file=data_root + 'cocodata/train_cocoformat.json'),
    test=dict(data_prefix=data_root + 'MOT17/test', ann_file=data_root + 'cocodata/test_cocoformat.json'))
total_epochs = 20
work_dir = 'workreiddir/'

