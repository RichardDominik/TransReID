#nohup python test.py --config_file configs/VeRi/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/vit_transreid_veri.pth' > ./logs/vit_eval.log 2>&1 &

destinationLogPath='./logs/vit_eval_trained_60_epochs.log'
trainedPth='./logs/veri_vit_transreid_stride/transformer_60.pth'
config='configs/VeRi/vit_transreid_stride.yml'

nohup python test.py --config_file $config  MODEL.DEVICE_ID "('0')" TEST.WEIGHT $trainedPth > $destinationLogPath 2>&1 &

