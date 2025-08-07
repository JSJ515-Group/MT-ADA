import torch
import torch.nn as nn

from actmtda.models.backbones import build_backbone
from actmtda.models.necks import build_neck
from actmtda.models.heads import build_head
from actmtda.utils.registry import RECOGNIZER


@RECOGNIZER.register("general_recognizer")
class GeneralRecognizer(nn.Module):
	def __init__(self, cfg):
		super(GeneralRecognizer, self).__init__()
		self.backbone = build_backbone(cfg.BACKBONE)
		if cfg.NECK.TYPE:
			self.neck = build_neck(cfg.NECK)
		self.head = build_head(cfg.HEAD)
		#self.rotation_head = nn.Linear(256, 4)  # 旋转角度预测类别为 4 (0°, 90°, 180°, 270°)


	def forward(self, x):
		feat = self.backbone(x)
		if hasattr(self, 'neck'):
			neck_feat = self.neck(feat)
			logits = self.head(neck_feat)
			return logits, neck_feat
		logits = self.head(feat)
		return logits, feat
	
	def encode_feat(self, x):
		feat = self.backbone(x)
		if hasattr(self, 'neck'):
			neck_feat = self.neck(feat)
			return neck_feat
		return feat
	
	def forward_classifier(self, feat, is_input_neck=False):
		if not is_input_neck and hasattr(self, 'neck'):
			feat = self.neck(feat)
		logits = self.head(feat)
		return logits

	def extract_low_level_features(self, x):
		"""
        提取低层次特征（经过 conv1 后传递给 layer1）
        """
		x = self.backbone.conv1(x)  # 首先通过 conv1 进行处理
		x = self.backbone.bn1(x)  # 执行批归一化
		x = self.backbone.relu(x)  # ReLU 激活
		x = self.backbone.maxpool(x)  # 最大池化
		low_level_feat = self.backbone.layer1(x)  # 提取低层次特征
		return low_level_feat

	def extract_high_level_features(self, x):
		"""
        提取高层次特征（经过 conv1 -> layer1 -> layer4）
        """
		x = self.backbone.conv1(x)  # 通过 conv1
		x = self.backbone.bn1(x)  # 执行批归一化
		x = self.backbone.relu(x)  # ReLU 激活
		x = self.backbone.maxpool(x)  # 最大池化
		x = self.backbone.layer1(x)  # 经过 layer1
		x = self.backbone.layer2(x)  # 经过 layer2
		x = self.backbone.layer3(x)  # 经过 layer3
		high_level_feat = self.backbone.layer4(x)  # 提取高层次特征
		return high_level_feat

	# def predict_rotation(self, x):
	# 	feat = self.backbone(x)  # 提取特征
	# 	if hasattr(self, 'neck'):
	# 		feat = self.neck(feat)  # 如果有 neck，传递给 neck
	# 	feat = feat.view(feat.size(0), -1)  # 展平特征
	# 	rotation_logits = self.rotation_head(feat)  # 通过旋转分类头预测旋转角度
	# 	return rotation_logits