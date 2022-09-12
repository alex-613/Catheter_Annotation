from PIL import Image
import numpy as np
import sys

im = Image.open("/home/atr17/PhD/Research/Phase_5_Al_model_building/DETR_net/CathNet/dataset_Catheter/Catheter_Simulations/Catheter_Simulation_001/us_simulation/masks/label001_catheter_908_083.png")

print(im.format, im.size, im.mode)

numpydata = np.asarray(im)
print(numpydata.shape)

np.set_printoptions(threshold=sys.maxsize)
print(np.max(numpydata))
print(type(numpydata))

catheter_mask = np.empty_like(numpydata)
aorta_mask = np.empty_like(numpydata)

catheter_mask = numpydata[numpydata==2]
aorta_mask = numpydata[numpydata==1]




