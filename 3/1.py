from abc import ABC, abstractmethod

import cv2
import numpy as np


def segment(x, y, ax, ay, bx, by):
    vx = x - ax
    vy = y - ay
    ux = bx - ax
    uy = by - ay

    t = (vx * ux + vy * uy) / (ux * ux + uy * uy)
    t = np.maximum(np.minimum(t, 1), 0)
    dx = vx - ux * t
    dy = vy - uy * t
    return np.hypot(dx, dy)


def minimum(a, b, c):
    return np.minimum(np.minimum(a, b), c)


class Shape(ABC):

    @abstractmethod
    def sdf(self, x, y):
        raise NotImplemented

    @abstractmethod
    def rgb(self, t, x, y):
        raise NotImplemented


class Circle(Shape):

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


class Plane(Shape):

    def __init__(self, x, y, nx, ny, r, b, g):
        self.x = x
        self.y = y
        self.nx = nx
        self.ny = ny
        self.r = r
        self.g = g
        self.b = b

    def sdf(self, x, y):
        return (x - self.x) * self.nx + (y - self.y) * self.ny

    def rgb(self, t, x, y):
        r = np.where(t, self.r, 0)
        g = np.where(t, self.g, 0)
        b = np.where(t, self.b, 0)
        return r, g, b


class Capsule(Shape):

    def __init__(self, ax, ay, bx, by, r, red, green, blue):
        self.ax = ax
        self.ay = ay
        self.bx = bx
        self.by = by
        self.r = r
        self.red = red
        self.green = green
        self.blue = blue

    def sdf(self, x, y):
        return segment(x, y, self.ax, self.ay, self.bx, self.by) - self.r

    def rgb(self, t, x, y):
        r = np.where(t, self.red, 0)
        g = np.where(t, self.green, 0)
        b = np.where(t, self.blue, 0)
        return r, g, b


class Rectangle(Shape):

    def __init__(self, cx, cy, theta, sx, sy, r, g, b):
        self.cx = cx
        self.cy = cy
        self.theta = theta
        self.sx = sx
        self.sy = sy
        self.r = r
        self.g = g
        self.b = b

    def sdf(self, x, y):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        dx = np.abs((x - self.cx) * c + (y - self.cy) * s) - self.sx
        dy = np.abs((y - self.cy) * c - (x - self.cx) * s) - self.sy
        ax = np.maximum(dx, 0)
        ay = np.maximum(dy, 0)
        return np.minimum(np.maximum(dx, dy), 0) + np.hypot(ax, ay)

    def rgb(self, t, x, y):
        r = np.where(t, self.r, 0)
        g = np.where(t, self.g, 0)
        b = np.where(t, self.b, 0)
        return r, g, b


class CornerRectangle(Shape):

    def __init__(self, cx, cy, theta, sx, sy, r, red, green, blue):
        self.cx = cx
        self.cy = cy
        self.theta = theta
        self.sx = sx
        self.sy = sy
        self.r = r
        self.red = red
        self.green = green
        self.blue = blue

    def sdf(self, x, y):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        dx = np.abs((x - self.cx) * c + (y - self.cy) * s) - self.sx
        dy = np.abs((y - self.cy) * c - (x - self.cx) * s) - self.sy
        ax = np.maximum(dx, 0)
        ay = np.maximum(dy, 0)
        return np.minimum(np.maximum(dx, dy), 0) + np.hypot(ax, ay) - self.r

    def rgb(self, t, x, y):
        r = np.where(t, self.red, 0)
        g = np.where(t, self.green, 0)
        b = np.where(t, self.blue, 0)
        return r, g, b


class Triangle(Shape):

    def __init__(self, ax, ay, bx, by, cx, cy, r, g, b):
        self.ax = ax
        self.ay = ay
        self.bx = bx
        self.by = by
        self.cx = cx
        self.cy = cy
        self.r = r
        self.g = g
        self.b = b

    def sdf(self, x, y):
        d0 = ((self.bx - self.ax) * (y - self.ay) > (self.by - self.ay) * (x - self.ax)) * \
             ((self.cx - self.bx) * (y - self.by) > (self.cy - self.by) * (x - self.bx)) * \
             ((self.ax - self.cx) * (y - self.cy) > (self.ay - self.cy) * (x - self.cx))
        d1 = np.minimum(
            segment(x, y, self.ax, self.ay, self.bx, self.by),
            segment(x, y, self.bx, self.by, self.cx, self.cy),
            segment(x, y, self.cx, self.cy, self.ax, self.ay),
        )
        return np.where(d0, -d1, d1)

    def rgb(self, t, x, y):
        r = np.where(t, self.r, 0)
        g = np.where(t, self.g, 0)
        b = np.where(t, self.b, 0)
        return r, g, b


class CornerTriangle(Shape):

    def __init__(self, ax, ay, bx, by, cx, cy, r, red, green, blue):
        self.ax = ax
        self.ay = ay
        self.bx = bx
        self.by = by
        self.cx = cx
        self.cy = cy
        self.r = r
        self.red = red
        self.green = green
        self.blue = blue

    def sdf(self, x, y):
        d0 = ((self.bx - self.ax) * (y - self.ay) > (self.by - self.ay) * (x - self.ax)) * \
             ((self.cx - self.bx) * (y - self.by) > (self.cy - self.by) * (x - self.bx)) * \
             ((self.ax - self.cx) * (y - self.cy) > (self.ay - self.cy) * (x - self.cx))
        d1 = minimum(
            segment(x, y, self.ax, self.ay, self.bx, self.by),
            segment(x, y, self.bx, self.by, self.cx, self.cy),
            segment(x, y, self.cx, self.cy, self.ax, self.ay),
        )
        return np.where(d0, -d1, d1) - self.r

    def rgb(self, t, x, y):
        r = np.where(t, self.red, 0)
        g = np.where(t, self.green, 0)
        b = np.where(t, self.blue, 0)
        return r, g, b


class Intersect(Shape):

    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

    def sdf(self, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        return np.where(sd1 < sd2, sd2, sd1)

    def rgb(self, t, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        rgb1 = self.shape1.rgb(t, x, y)
        rgb2 = self.shape2.rgb(t, x, y)
        return np.where(sd1 < sd2, rgb2, rgb1)


class Union(Shape):

    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

    def sdf(self, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        return np.where(sd1 < sd2, sd1, sd2)

    def rgb(self, t, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        rgb1 = self.shape1.rgb(t, x, y)
        rgb2 = self.shape2.rgb(t, x, y)
        return np.where(sd1 < sd2, rgb1, rgb2)


class Subtract(Shape):

    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

    def sdf(self, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        return np.where(sd1 < -sd2, -sd2, sd1)

    def rgb(self, t, x, y):
        return self.shape1.rgb(t, x, y)


x = np.tile(np.arange(0, 500), (500, 1)).T
y = np.transpose(x)
x = np.repeat(x[:, :, np.newaxis], 64, axis=2)
y = np.repeat(y[:, :, np.newaxis], 64, axis=2)
# theta = 2 * np.pi * np.random.random((500, 500, 64))
# theta = 2 * np.pi / 64 * np.tile(np.arange(0, 64), (500, 500, 1))
theta = 2 * np.pi / 64 * (np.tile(np.arange(0, 64), (500, 500, 1)) + np.random.random((500, 500, 64)))
c = np.cos(theta)
s = np.sin(theta)

# 1
# circle = Circle(250, 250, 80, 255, 255, 255)
# plane = Plane(250, 250, 1, 0, 200, 200, 200)
# shape = Intersect(circle, plane)

# 2
# shape = Capsule(150, 150, 350, 350, 50, 255, 255, 255)

# 3
# shape = Rectangle(250, 250, -2 * np.pi / 16, 50, 150, 255, 255, 255)

# 4
# shape = CornerRectangle(250, 250, -2 * np.pi / 16, 50, 150, 50, 255, 255, 255)

# 5
# shape = Triangle(150, 250, 350, 150, 450, 350, 255, 255, 255)

# 6
shape = CornerTriangle(100, 250, 300, 150, 400, 400, 50, 255, 255, 255)

step = 64
epsilon = 1e-6
t0 = np.zeros((500, 500, 64), dtype=np.bool)
t1 = np.zeros((500, 500, 64))
for i in range(step):
    print(i)
    sd = np.where(t0, 0, shape.sdf(x + c * t1, y + s * t1))
    t0 = np.where(t0, True, sd < epsilon)
    t1 = np.where(t0, t1, t1 + sd)

r, g, b = shape.rgb(t0, x + c * t1, y + s * t1)
r = 1 / 64 * np.sum(r, axis=2)
g = 1 / 64 * np.sum(g, axis=2)
b = 1 / 64 * np.sum(b, axis=2)

image = np.stack([r, g, b], axis=2)
image = np.minimum(image, 255)
image = image.astype(np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("image", image)
cv2.waitKey(0)
