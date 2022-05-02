destinationLogPath='./logs/swin_vis_120_epochs.log'
trainedPth='./logs/veri_vit_transreid_stride/transformer_120.pth'
config='configs/VeRi/vit_transreid_stride.yml'

nohup python get_vis_results.py --config_file $config MODEL.DEVICE_ID "('0')" TEST.WEIGHT $trainedPth > $destinationLogPath 2>&1 &
