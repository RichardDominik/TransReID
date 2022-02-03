# https://github.com/lulujianjie/person-reid-tiny-baseline/blob/master/tools/get_vis_result.py

import os
import argparse
import sys
from config import cfg
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
sys.path.append('.')
from utils.logger import setup_logger
from model import make_model
from datasets import make_dataloader
import numpy as np
import cv2
from utils.metrics import cosine_similarity


def visualizer(test_img, camid, top_k = 10, img_size=[128,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        img = np.asarray(Image.open('./data/VeRi/image_test/' + img_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        title=name
    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    if not os.path.exists(cfg.LOG_DIR+ "/results/"):
        print('need to create a new folder named results in {}'.format(Cfg.LOG_DIR))
    cv2.imwrite(Cfg.LOG_DIR+ "/results/{}-cam{}.png".format(test_img,camid),figure)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="ReID Baseline Visualization")
	parser.add_argument(
	"--config_file", default="", help="path to config file", type=str
	)
	parser.add_argument("opts", help="Modify config options using the command-line", default=None,
				nargs=argparse.REMAINDER)

	args = parser.parse_args()

	if args.config_file != "":
		cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()

	os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
	cudnn.benchmark = True

	train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

	model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
	model.load_param(cfg.TEST.WEIGHT)

	device = 'cuda'
	model = model.to(device)

	transform = T.Compose([
		T.Resize(cfg.INPUT.SIZE_TEST),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
	log_dir = cfg.LOG_DIR
	logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), log_dir, if_train=False)
	model.eval()
	for test_img in os.listdir(cfg.QUERY_DIR):
		logger.info('Finding ID {} ...'.format(test_img))
		
		gallery_feats = torch.load(cfg.LOG_DIR + '/gfeats.pth')

		img_path = np.load('./logs/imgpath.npy')
		print(gallery_feats.shape, len(img_path))
		query_img = Image.open(cfg.QUERY_DIR + '/' + test_img)
		input = torch.unsqueeze(transform(query_img), 0)
		input = input.to(device)
		with torch.no_grad():
			query_feat = model(input)

		dist_mat = cosine_similarity(query_feat, gallery_feats)
		indices = np.argsort(dist_mat, axis=1)
		visualizer(test_img, camid='mixed', top_k=10, img_size=cfg.INPUT.SIZE_TEST)
