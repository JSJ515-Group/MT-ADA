import torch
import torchvision.transforms.functional as TF

def apply_random_rotation(images):
    """
    对输入的图像随机旋转并生成旋转标签。
    :param images: 输入的图像批次，Tensor格式
    :return: 旋转后的图像和相应的旋转标签
    """
    rotated_images = []
    rotation_labels = []
    rotation_angles = [0, 90, 180, 270]  # 四个旋转角度
    for img in images:
        angle = rotation_angles[torch.randint(0, 4, (1,)).item()]  # 随机选择一个角度
        rotated_images.append(TF.rotate(img, angle))  # 旋转图像
        rotation_labels.append(rotation_angles.index(angle))  # 对应的旋转标签

    rotated_images = torch.stack(rotated_images)  # 将旋转后的图像合并为一个批次
    rotation_labels = torch.tensor(rotation_labels).long()  # 转换为张量
    return rotated_images, rotation_labels
