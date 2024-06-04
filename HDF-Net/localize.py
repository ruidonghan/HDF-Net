import os
import cv2
import torch
from torchvision.transforms import transforms as T
from net import HDF_Net
from dataset import LiverDataset
from torch.utils.data import DataLoader

# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    # T.ToPILImage(),
    T.Resize([256,256]),
    # T.RandomHorizontalFlip(),  # 随机水平翻转
    # T.RandomRotation(degrees=15),  # 随机旋转
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# mask只需要转换为tensor
y_transform = T.Compose([
    T.Resize([256,256]),
    T.ToTensor(),
])

# 测试
def test():
    model = HDF_Net().to(device)
    model.load_state_dict(torch.load("./casia_train/weights_100.pth", map_location='cpu'))
    liver_dataset = LiverDataset(r"F:\ImageManipulationDatasets\test-casia1\test", transform=x_transform,
                                 target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset, drop_last=True)  # batch_size默认为1
    model.eval()
    with torch.no_grad():
        i=0
        for x,_ in dataloaders:
            x=x.to(device)
            y1 = model(x)
            y1 = (torch.sigmoid(y1[0, 0]) * 255).cpu().numpy()    # 灰度图
            # y2 = (torch.sigmoid(y2[0, 0]) * 255).cpu().numpy()
            # y3 = (torch.sigmoid(y3[0, 0]) * 255).cpu().numpy()
            _, y1=cv2.threshold(y1,127,255,cv2.THRESH_BINARY)     # 二值图
            # _, y2 = cv2.threshold(y2, 127, 255, cv2.THRESH_BINARY)
            # _, y3 = cv2.threshold(y3, 127, 255, cv2.THRESH_BINARY)
            save_path='./casia/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            i+=1
            cv2.imwrite(save_path+'/'+str(i)+'_mask'+'.png', y1)
            # cv2.imshow('hrd',y)
            # cv2.waitKey()
        #     img_y = torch.squeeze(y).numpy()
        #     plt.imshow(img_y)
        #     plt.pause(0.01)
        # plt.show()


if __name__ == '__main__':

    test()