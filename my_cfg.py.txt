_base_ = [
    '/mmocr/configs/_base_/recog_models/sar.py',
    '/mmocr/configs/_base_/recog_pipelines/sar_pipeline.py',
    '/mmocr/configs/_base_/schedules/schedule_adam_step_5e.py',
    '/mmocr/configs/_base_/default_runtime.py',
]

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}


work_dir = '/mmocr/demo/tutorial_exps'
dataset_type = 'OCRDataset'
root = '/mmocr/tests/data/toy_dataset' # Location where the annotation and crop images are being stored

img_prefix ='/mmocr/tests/data/toy_dataset/crops'
train_anno_file1 = '/mmocr/tests/data/toy_dataset/train_label.jsonl'


loader_dt_train = dict(type='AnnFileLoader',
                            repeat=100,                   
                            file_format='txt',  # only txt and lmdb
                            file_storage_backend='disk',
                            parser=dict(type='LineJsonParser',
                                        keys=['filename', 'text']))

train_datasets1 = dict(type='OCRDataset',
                       img_prefix=img_prefix,
                       ann_file=train_anno_file1,
                       loader=loader_dt_train,
                       pipeline=None,           
                       test_mode=False)

loader_dt_val = dict(type='AnnFileLoader',
                            repeat=1,                   
                            file_format='txt',  # only txt and lmdb
                            file_storage_backend='disk',
                            parser=dict(type='LineJsonParser',
                                        keys=['filename', 'text']))

val_datasets1 = dict(type='OCRDataset',
                       img_prefix=img_prefix,
                       ann_file=train_anno_file1,
                       loader=loader_dt_val,
                       pipeline=None,           
                       test_mode=False)




train_list = [train_datasets1]
test_list = [val_datasets1]


data = dict(
    workers_per_gpu=2,
    samples_per_gpu=8,
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline)
    )

evaluation = dict(interval=1, metric='acc')