destinationLogPath='./logs/swin_eval_trained_60_epochs.log'
trainedPth='./logs/veri_swin_transreid/transformer_60.pth'
config='configs/VeRi/swin_transformer_transreid.yml'

nohup python test.py --config_file $config  MODEL.DEVICE_ID "('0')" TEST.WEIGHT $trainedPth > $destinationLogPath 2>&1 &