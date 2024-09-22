# Structure
![image](https://github.com/dadaguailhb/TCL/blob/main/img/motivation1.png)
![image](https://github.com/dadaguailhb/TCL/blob/main/img/%E6%A1%86%E6%9E%B6_%E5%89%AF%E6%9C%AC2.png)


## Datasets
[JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
[PIE](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)

## Requirements
Installing requirements:
```bash
pip install -r requirements.txt
```

## Training
Training on JAAD
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

Training on PIE
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
