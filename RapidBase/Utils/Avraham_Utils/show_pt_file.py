import matplotlib.pyplot as plt
import sys, torch


img_name = sys.argv[1]
plt.imshow(torch.load(img_name).cpu().squeeze().numpy())
plt.show()
