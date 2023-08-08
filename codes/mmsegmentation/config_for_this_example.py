
# Train Segformer Mit B0
_base_ = [
    #"./configs/_base_/models/segformer_mit-b0.py",
    "./configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-512x512.py",
    "dataset_setting.py",
    "./configs/_base_/default_runtime.py",
    #"./configs/_base_/schedules/schedule_160k.py"
]
# mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    #mean=[0., 0., 0.],
    #std=[255., 255., 255.],
    bgr_to_rgb=True,
    size=(1024, 1024),
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type='EncoderDecoderWithoutArgmax',
    init_cfg=dict(
        type='Pretrained',
        # load ADE20k pretrained EncoderDecoder from mmsegmentation
        checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth"
        #checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth"
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='SegformerHeadWithoutAccuracy',
        num_classes=29,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)

# optimizer
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
#optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)
