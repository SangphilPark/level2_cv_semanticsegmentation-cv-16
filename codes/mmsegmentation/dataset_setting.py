
# dataset settings
dataset_type = 'XRayDataset'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadXRayAnnotations'),
    dict(type='Resize', scale=(1024, 1024)),
    #dict(type='RandomRotate', prob=0.5, degree=90.),
    #dict(type='BioMedicalGaussianBlur'),
    dict(type='CLAHE', clip_limit=1.0, tile_grid_size=(8, 8)),
    dict(type='RandomCutOut', prob=0.5, n_holes=(20, 40), cutout_ratio=(0.2, 0.2)),
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadXRayAnnotations'),
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024)),
    #dict(type='BioMedicalGaussianBlur'),
    dict(type='CLAHE', clip_limit=1.0, tile_grid_size=(8, 8)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        is_train=True,
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        is_train=False,
        pipeline=val_pipeline
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        is_train=False,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='DiceMetric')
test_evaluator = val_evaluator
