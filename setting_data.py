# testのファイル名は0から始まっているのに対し、trainのファイル名は1から始まっていて
# データセットを作る時不都合なのでtrainを0.pngから始まるように直す

import os

dir_fullpath = os.path.dirname(__file__)

train_dir = dir_fullpath + '/Segmentation01/train'
image_dir = train_dir + '/org'
label_dir = train_dir + '/label'

# 実行済みであれば処理を行わない分岐
if os.path.isfile(image_dir + '/0.png'):
    pass

else:
    num_img = len(os.listdir(image_dir))

    for n in range(num_img):
        old_image_path = os.path.join(image_dir, str(n + 1) + '.png')
        new_image_path = os.path.join(image_dir, str(n) + '.png')
        os.rename(old_image_path, new_image_path)
        old_label_path = os.path.join(label_dir, str(n + 1) + '.png')
        new_label_path = os.path.join(label_dir, str(n) + '.png')
        os.rename(old_label_path, new_label_path)