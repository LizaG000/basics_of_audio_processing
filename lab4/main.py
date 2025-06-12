import cv2
import numpy as np
import matplotlib.pyplot as plt

# Фильтр резкости, но я игралась со свето тенью
image = cv2.imread('288af49966e71c911210cbe91107695d.jpg')
kernel = np.array([[2, 1, -2],
                  [4, -2, -3],
                  [2, 1, -2]])
filtered_image = cv2.filter2D(image, -10, kernel)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb_filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title('Исходное изображение')
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(rgb_filtered_image)
plt.title('Фильтрованное изображение')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
fig.savefig('фильтр резкости.png')
plt.close(fig)

# Размытие по гауссу
image = cv2.imread('c94eb5f87a8c96c83c146a8cb543a28b.jpg')
kernel_size = (50, 50)
sigma = 0.0
kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
gaussian_kernel = np.outer(kernel, kernel)
blurred_image = cv2.filter2D(image, -1, gaussian_kernel)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb_filtered_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title('Исходное изображение')
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(rgb_filtered_image)
plt.title('Фильтрованное изображение')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
fig.savefig('гауссовское размытие.png')
plt.close(fig)

# Фильтр выявления ребер
image = cv2.imread('3a21afc485b711fdc15889c948ecffab.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel_x = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
filtered_image_x = cv2.filter2D(gray_image, -1, sobel_x)
sobel_y = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])
filtered_image_y = cv2.filter2D(gray_image, -1, sobel_y)
edge_image = cv2.addWeighted(filtered_image_x, 0.5, filtered_image_y, 0.5, 0)

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title('Исходное изображение')
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(edge_image, cmap='gray')
plt.title('Фильтрованное изображение')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
fig.savefig('фильтр выявления ребер.png')
plt.close(fig)



# Фильтрация верхних частот в изображении
img = cv2.imread('3a34d3fbc1c01a941f07dd0a9f88743f.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
mask = np.ones((rows, cols), np.uint8)
r = 40
mask[crow - r:crow + r, ccol - r:ccol + r] = 0
fshift = fshift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.abs(np.fft.ifft2(f_ishift))
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(img_back, cmap='gray')
plt.title('Фильтрованное изображение')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
fig.savefig('верхние частоты.png')
plt.close(fig)

# Фильтрация низких частот
img = cv2.imread('3a34d3fbc1c01a941f07dd0a9f88743f.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
mask = np.zeros((rows, cols), np.uint8)
r = 35
mask[crow - r:crow + r, ccol - r:ccol + r] = 1
fshift = fshift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.abs(np.fft.ifft2(f_ishift))
fig = plt.figure(figsize=(12, 6), dpi=100)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Исходное изображение')
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(img_back, cmap='gray')
plt.title('Фильтрованное изображение')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
fig.savefig('низкие частоты.png')
plt.close(fig)