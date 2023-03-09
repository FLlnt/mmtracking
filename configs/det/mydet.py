USE_MMDET = True
_base_ = ['./faster-rcnn_r50_fpn_4e_mot17-half.py']
# data
data_root = '/Volumes/2019_6/冯泽霖/飞虱/'
data = dict(
    train=dict(ann_file=data_root + 'cocodata/train_cocoformat.json'),
    val=dict(ann_file=data_root + 'cocodata/train_cocoformat.json'),
    test=dict(ann_file=data_root + 'cocodata/test_cocoformat.json'))
total_epochs = 20
work_dir = 'workdetdir/'
resume_from = 'workdetdir/latest.pth'
