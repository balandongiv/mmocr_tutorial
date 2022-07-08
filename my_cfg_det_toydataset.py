_base_ = [
    '/mmocr/configs/_base_/default_runtime.py',
    '/mmocr/configs/_base_/schedules/schedule_sgd_1200e.py',
    '/mmocr/configs/_base_/det_models/dbnetpp_r50dcnv2_fpnc.py',
    '/mmocr/configs/_base_/det_pipelines/dbnet_pipeline.py',
]
# /mmocr/configs/_base_/default_runtime.py
# The YML suggested the DBNetpp is with Training Resources: 1x Nvidia A100
# Location where the annotation and crop images are being stored
root='/content/wdr'


# Set up working dir to save files and logs.
work_dir =f'{root}/train_detect/base_dbnetpp'


train_root_custm1 ='/mmocr/tests/data/toy_dataset/imgs/test'

train_custm1 = dict(            # This is the new one by
    type='TextDetDataset',
    img_prefix=train_root_custm1,
    ann_file='/content/instances_training.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=300,
        file_format='txt',
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=None,
    test_mode=False)


val_custm1 = dict(            # This is the new one by
    type='TextDetDataset',
    img_prefix=train_root_custm1,
    ann_file=f'/content/instances_training.txt',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=None,
    test_mode=False)




train_list = [train_custm1]
test_list = [val_custm1]



train_pipeline_r50dcnv2 = {{_base_.train_pipeline_r50dcnv2}}
test_pipeline_4068_1024 = {{_base_.test_pipeline_4068_1024}}


data = dict(
    samples_per_gpu=16, # Default 32
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r50dcnv2),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024))



evaluation = dict(
    interval=20,
    metric='hmean-iou')


