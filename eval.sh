#nohup python test.py --config_file configs/VeRi/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/vit_transreid_veri.pth' > ./logs/vit_eval.log 2>&1 &

nohup python test.py --config_file configs/VeRi/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/veri_vit_transreid_stride/transformer_120.pth' > ./logs/vit_eval_trained.log 2>&1 &

