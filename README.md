# TCL

## train on JAAD

```bash
OMP_NUM_THREADS=1 nohup python -m torch.distributed.launch --nproc_per_node=6 \
       --nnodes=1 \
      --node_rank=0  \
      tools/train_ddp_videomae.py \
      --config_file configs/JAAD_intent.yaml \
      STYLE SF-GRU \
      MODEL.WITH_TRAFFIC True \
      SOLVER.MAX_ITERS 15000 \
      TEST.BATCH_SIZE 4 \
      SOLVER.SCHEDULER none \
      DATASET.BALANCE False \
      MODEL.WITH_POSE True \
      MODEL.TASK intent_single
```

## train on PIE
```bash
OMP_NUM_THREADS=1 nohup python -m torch.distributed.launch --nproc_per_node=6 \
       --nnodes=1 \
      --node_rank=0  \
      tools/train_ddp_videomae.py \
      --config_file configs/PIE_intent.yaml \
      STYLE SF-GRU \
      MODEL.WITH_TRAFFIC True \
      SOLVER.MAX_ITERS 15000 
      TEST.BATCH_SIZE 4 \
      SOLVER.SCHEDULER none \
      DATASET.BALANCE False \
      MODEL.WITH_POSE True \
      MODEL.TASK intent_single
