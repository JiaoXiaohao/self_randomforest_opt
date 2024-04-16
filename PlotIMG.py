from osgeo import gdal
import os
import matplotlib.pyplot as plt
import numpy as np
IMGPATH = "data/img.tif"
data = gdal.Open(IMGPATH)
dataArray = data.ReadAsArray()
print(dataArray.shape)
# print(plt.style.available)
plt.style.use("ggplot")  # 绘制遥感影像
plt.figure(figsize=(8, 8))
# 绘制4，3，2波段
plt.imshow(np.transpose(dataArray[[4, 3, 2], :, :]))
plt.axis("off")
plt.savefig("img.png", dpi=300, bbox_inches="tight")
plt.show()
