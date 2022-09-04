import os 
import random
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np




images_dir =os.path.join(os.getcwd(),"archive/images")
segmaps_dir =os.path.join(os.getcwd(),"archive/segmaps")

fns =os.listdir(images_dir)
fns = [fn.split('.')[0] for fn in fns]


fig, axes = plt.subplots(3, 2, figsize=(16, 8))
#axes is of shape 3x2 and .ravel() flattened the axes to 1x6
axes = axes.ravel() 

for ax in axes:
    ax.axis('off')
for i in range(0, 6, 2):
    fn = f'{random.sample(fns, 1)[0]}.png'
    image = cv.imread(os.path.join(images_dir, fn),0)
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    segmap = cv.imread(os.path.join(segmaps_dir, fn),cv.IMREAD_UNCHANGED)
    axes[i].imshow(image)
    axes[i+1].matshow(segmap, cmap='tab20')
plt.show()