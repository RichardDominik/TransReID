destinationLogPath='./logs/swin_vis_120_epochs.log'
trainedPth='./logs/veri_swin_transformer_v2/swin_transformer_120.pth'
config='configs/VeRi/swin_transformer_v2.yml'

#nohup python visualize.py --config_file $config MODEL.DEVICE_ID "('0')" TEST.WEIGHT $trainedPth > $destinationLogPath 2>&1 &

nohup python tools/get_vis_results.py --config_file $config MODEL.DEVICE_ID "('0')" TEST.WEIGHT $trainedPth > $destinationLogPath 2>&1 &