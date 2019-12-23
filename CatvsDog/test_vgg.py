import os, sys
import torch
import cv2
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image

def pictureclassify(picture):
    classes = ('cat', 'dog')
    """将上传的图片进行图片识别分类"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('CatvsDog/modelcatdog.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式

    img = cv2.imread(picture.photo.path)  # 读取要预测的图片
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    output = model(img)
    prob = F.softmax(output)  # prob是2个分类的概率
    #print(prob)
    value, predicted = torch.max(output.data, 1)
    #print(predicted.item())
    #print(value)
    pred_class = classes[predicted.item()]
    #print(pred_class)
    if pred_class=='cat':
        label="猫"
    else:
        label='狗'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

    return label
