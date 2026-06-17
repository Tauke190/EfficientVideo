
checkpoint_path="/home/av354855/EfficientVideo/checkpoints/saved_model/LVU/relationship/checkpoint_best.pth"

python -m torch.distributed.run --nproc_per_node=1 \
    --master_port=34650 \
    train.py \
    --cfg-path lavis/projects/malmm/cls_breakfast.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned True \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 20 \
    model.num_frames 100 \
    run.init_lr 1e-4 \
    run.max_epoch 20 \
    run.num_beams 5 \
    run.batch_size_train 16 \
    run.batch_size_eval 16 \
    run.accum_grad_iters 1 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate True \
    run.report_metric True \
    run.prefix test \
    run.resume_ckpt_path ${checkpoint_path}

