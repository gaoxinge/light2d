import cv2
import numpy as np


class Circle:

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def sdf(self, x, y):
        return np.hypot(x - self.x, y - self.y) - self.r


x = np.tile(np.arange(0, 500), (500, 1)).T
y = np.transpose(x)
x = np.repeat(x[:, :, np.newaxis], 64, axis=2)
y = np.repeat(y[:, :, np.newaxis], 64, axis=2)
# theta = 2 * np.pi * np.random.random((500, 500, 64))
# theta = 2 * np.pi / 64 * np.tile(np.arange(0, 64), (500, 500, 1))
theta = 2 * np.pi / 64 * (np.tile(np.arange(0, 64), (500, 500, 1)) + np.random.random((500, 500, 64)))
c = np.cos(theta)
s = np.sin(theta)

circle = Circle(250, 250, 50)

step = 3
epsilon = 1e-3
t = np.zeros((500, 500, 64))
sd = circle.sdf(x + c * t, y + s * t)
v = circle.sdf(x + c * t, y + s * t)
for i in range(step):
    t += sd
    sd = circle.sdf(x + c * t, y + s * t)
    v = np.minimum(v, sd)

v = v < epsilon
tmp = np.ones((500, 500, 64)) * 150
tmp = v * tmp

image = 2 * np.pi / 64 * np.sum(tmp, axis=2)
image = np.minimum(image, 255)
image = image.astype(np.uint8)
image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("image", image)
cv2.waitKey(0)
