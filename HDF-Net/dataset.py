import torch.utils.data as data
import os
import PIL.Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# data.Dataset:
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)

class LiverDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None, target_transform=None):  # root表示图片路径
        n = len(os.listdir(root)) // 2  # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整

        imgs = []
        for i in range(n):
            img = os.path.join(root, "%d_rgb.jpg" % (i+1))  # os.path.join(path1[,path2[,......]]):将多个路径组合后返回
            mask = os.path.join(root, "%d_mask.jpg" % (i+1))
            imgs.append([img, mask])  # append只能有一个参数，加上[]变成一个list

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert("RGB")
        img_y = Image.open(y_path)
        #noise  2023.6.29
        # img_arr=np.array(img_x)
        # noise=np.random.normal(0,15,img_arr.shape)
        # img_arr=img_arr+noise
        # img_arr=np.clip(img_arr,0,255).astype(np.uint8)
        # noise_img=Image.fromarray(img_arr)

        #gaussian blur 2023.6.29
        # img=cv2.imread(x_path)
        # img=cv2.GaussianBlur(img,(5,5),1)

        #JPEG Compression
        # jpeg_img= self.compress_image(img_x,quality=75)

        if self.transform is not None:
            img_x = self.transform(img_x)
            # noise_img=self.transform(noise_img)   # noise 2023.6.29
            # img=self.transform(img)
            # jpeg_img=self.transform(jpeg_img)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
            # img_z = self.target_transform(img_z)
        return img_x, img_y  # 返回的是图片

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有三个元素，[img,mask, edge]

    # def compress_image(self,image,quality=100):
    #     self.image=image
    #     self.quality=quality
    #     img_bytes=io.BytesIO()
    #     image.save(img_bytes,format='JPEG',quality=quality)
    #     img_array=np.frombuffer(img_bytes.getvalue(),dtype=np.uint8)
    #     jpeg_img=Image.open(io.BytesIO(img_array))
    #     return jpeg_img

