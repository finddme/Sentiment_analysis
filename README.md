# TE(Text Empathy) 3.0
- Emotion class를 예측하는 KoELECTRA기반 classification fine-tune task

## model information
- KoElectra(https://github.com/monologg/KoELECTRA/tree/024fbdd600e653b6e4bdfc64ceec84181b5ce6c4)
- version: KoELECTRA-Base-v3

## Environment
- ubuntu 20.04
- python 3.9.12
- docker image
```
docker pull ayaanayaan/ayaan_nv
```

## Requirements
- pytorch 1.10
- pymongo 4.1.1


## run_classify.py(Koelectra)
```
# Train
python main.py --op train --target_gpu (0/1/2) --ck_path (ck_path)

# Train with multi gpu
python main.py --op train --target_gpu m --ck_path (ck_path)

# Test
python main.py --op test --target_gpu (0/1/2) --load_ck (ck_path)

```

## run_bert.py(Kobert)
```
# Train
python main.py --op train_bert --target_gpu (0/1/2) --ck_path (ck_path)

# Train with multi gpu
python main.py --op train_bert --target_gpu m --ck_path (ck_path)

# Test
python main.py --op test_bert --target_gpu (0/1/2) --load_ck (ck_path)

```

## api.py(Koelectra)
```
python main.py --op api --target_gpu (0/1/2) --load_ck (ck_path) --port (port)
```
