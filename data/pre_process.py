import glob
import os
from PIL import Image
import io
def resizing(path, dst_size):
    try:
        img = Image.open(path)
    except:
        img = Image.open(io.BytesIO(path))
    img_size = img.size #(가로,세로)
    if img_size.index(min(img_size))==0:#높은 이미지
        resized_img = img.resize(
                                (dst_size,
                                (height:=int((dst_size/img_size[0])*img_size[1])))
                                )
        resized_img = resized_img.crop((0,
                                       up:=int((height-(dst_size*(4/3)))/2),
                                       480,
                                       int(up+dst_size*(4/3)))
        )

                         
    else:#넓은 이미지
        resized_img = img.resize(
                                ((width:=int((dst_size/img_size[1])*img_size[0])),
                                dst_size)
                                )
        resized_img = resized_img.crop((left:=int((width-(dst_size*(4/3)))/2),
                                       0,
                                       int(left+dst_size*(4/3)),
                                       480
                                       ))
    return resized_img

def make_dst_path(src_dir, dst_dir,path):
    dst_path = path.replace(src_dir, dst_dir)
    return dst_path

if __name__ =='__main__':
    src_dir = r''
    dst_dir = r''
    dst_size = 480
    imgs = glob.glob(src_dir + '/*/*')
    for img in imgs:
        resized_img = resizing(img, dst_size)
        dst_path = make_dst_path(src_dir, dst_dir,img)
        resized_img.save(dst_path)
