import cv2
import numpy as np


class Circle:

    def __init__(self, x, y, r, red, green, blue):
        self.x = x
        self.y = y
        self.r = r
        self.red = red
        self.green = green
        self.blue = blue

    def sdf(self, x, y):
        return np.hypot(x - self.x, y - self.y) - self.r

    def rgb(self, t, x, y):
        r = np.where(t, self.red, 0)
        g = np.where(t, self.green, 0)
        b = np.where(t, self.blue, 0)
        return r, g, b


x = np.tile(np.arange(0, 500), (500, 1)).T
y = np.transpose(x)
x = np.repeat(x[:, :, np.newaxis], 64, axis=2)
y = np.repeat(y[:, :, np.newaxis], 64, axis=2)
# theta = 2 * np.pi * np.random.random((500, 500, 64))
# theta = 2 * np.pi / 64 * np.tile(np.arange(0, 64), (500, 500, 1))
theta = 2 * np.pi / 64 * (np.tile(np.arange(0, 64), (500, 500, 1)) + np.random.random((500, 500, 64)))
c = np.cos(theta)
s = np.sin(theta)

circle = Circle(250, 250, 50, 255, 255, 255)

step = 10
epsilon = 1e-6
t0 = np.zeros((500, 500, 64), dtype=np.bool)
t1 = np.zeros((500, 500, 64))
for i in range(step):
    print(i)
    sd = np.where(t0, 0, circle.sdf(x + c * t1, y + s * t1))
    t0 = np.where(t0, True, sd < epsilon)
    t1 = np.where(t0, t1, t1 + sd)

r, g, b = circle.rgb(t0, x + c * t1, y + s * t1)
r = 1 / 64 * np.sum(r, axis=2)
g = 1 / 64 * np.sum(g, axis=2)
b = 1 / 64 * np.sum(b, axis=2)

image = np.stack([r, g, b], axis=2)
image = np.minimum(image, 255)
image = image.astype(np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("image", image)
cv2.waitKey(0)
