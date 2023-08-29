import glob
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
class CowDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train', val_ratio=0.2, seed = 42):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.val_ratio = val_ratio
        self.seed = 42
        self.img_paths = []
        self.setup()

    def setup(self):
        random.seed(self.seed)
        cow_dirs = glob.glob(self.data_dir + '/*')
        for cow_dir in cow_dirs:
            cows = glob.glob(cow_dir + '/*')
            random.shuffle(cows)
            cows_len = len(cows)
            if self.mode == 'train':
                self.img_paths += cows[0:int((1-self.val_ratio)*cows_len)]
            elif self.mode =='val':
                self.img_paths += cows[int((1-self.val_ratio)*cows_len):]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        transformed_img = self.transform(img)
        
        if '1++' in img_path:
            label = 0
        elif '1+' in img_path:
            label = 1
        elif '2' in img_path:
            label = 2
        elif '3' in img_path:
            label = 3
        return transformed_img, label
'''
    augmentation
'''
train_transform = transforms.Compose([transforms.RandomCrop((480,480)),
                                      transforms.RandomApply(
                                          [transforms.RandomRotation((89,90))],
                                      0.5),
                                      transforms.RandomVerticalFlip(0.5),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor(),
                                      transforms.ColorJitter((1,1.2))
                                      ])

val_transform = transforms.Compose([transforms.CenterCrop((480,480)),
                                      transforms.ToTensor()])

if __name__=='__main__':
    dataset = CowDataset(r'C:\Users\hyungu_lee\Downloads\축산물 품질(QC) 이미지\Training\새 폴더', transform=train_transform)
    print(dataset[0])
    print(dataset[0][0].size())
    plt.imshow(transforms.functional.to_pil_image(dataset[0][0]))
    plt.show()
                                      