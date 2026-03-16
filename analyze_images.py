from imaris_ims_file_reader.ims import ims
from matplotlib import pyplot as plt
import numpy as np

def get_mip_image(a, channel):

    mip = np.max(a[0, channel, :, :, :], axis=0)

    print(mip.shape)
    return mip

file = r'..\data\2-13-26 GA rapamycin\2026-02-13\control_1.ims'

a = ims(file)

# (t,c,z,y,x) = (1, 2, 16, 512, 2048)
mip_ch0 = get_mip_image(a, 0)

plt.imshow(mip_ch0)
plt.show()

mip_ch1 = get_mip_image(a, 1)

plt.imshow(mip_ch1)
plt.show()
# print(a.ResolutionLevelLock)
# print(a.ResolutionLevels)
# print(a.TimePoints)
# print(a.Channels)
# print(a.shape)
# print(a.chunks)
# print(a.dtype)
# print(a.ndim)