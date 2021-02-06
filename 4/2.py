import time
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


def gradient(shape, x, y, epsilon):
    xph = shape.sdf(x + epsilon, y)
    xmh = shape.sdf(x - epsilon, y)
    yph = shape.sdf(x, y + epsilon)
    ymh = shape.sdf(x, y - epsilon)
    return (xph - xmh) / (2 * epsilon), (yph - ymh) / (2 * epsilon)


def reflect(ix, iy, nx, ny):
    dot = 2 * (ix * nx + iy * ny)
    return ix - dot * nx, iy - dot * ny


class Shape(ABC):

    @abstractmethod
    def sdf(self, x, y):
        raise NotImplemented

    @abstractmethod
    def rgb(self, x, y):
        raise NotImplemented

    @abstractmethod
    def reflectivity(self, x, y):
        raise NotImplemented


class Corner(Shape):

    def __init__(self, shape, r):
        self.shape = shape
        self.r = r

    def sdf(self, x, y):
        return self.shape.sdf(x, y) - self.r

    def rgb(self, x, y):
        return self.shape.rgb(x, y)

    def reflectivity(self, x, y):
        return self.shape.reflectivity(x, y)


class Intersect(Shape):

    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

        self.tmp = None

    def sdf(self, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        self.tmp = sd1 < sd2
        return np.where(self.tmp, sd2, sd1)

    def rgb(self, x, y):
        r1, g1, b1 = self.shape1.rgb(x, y)
        r2, g2, b2 = self.shape2.rgb(x, y)
        r = np.where(self.tmp, r2, r1)
        g = np.where(self.tmp, g2, g1)
        b = np.where(self.tmp, b2, b1)
        return r, g, b

    def reflectivity(self, x, y):
        reflectively1 = self.shape1.reflectivity(x, y)
        reflectively2 = self.shape2.reflectivity(x, y)
        return np.where(self.tmp, reflectively2, reflectively1)


class Union(Shape):

    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

        self.tmp = None

    def sdf(self, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        self.tmp = sd1 < sd2
        return np.where(self.tmp, sd1, sd2)

    def rgb(self, x, y):
        r1, g1, b1 = self.shape1.rgb(x, y)
        r2, g2, b2 = self.shape2.rgb(x, y)
        r = np.where(self.tmp, r1, r2)
        g = np.where(self.tmp, g1, g2)
        b = np.where(self.tmp, b1, b2)
        return r, g, b

    def reflectivity(self, x, y):
        reflectively1 = self.shape1.reflectivity(x, y)
        reflectively2 = self.shape2.reflectivity(x, y)
        return np.where(self.tmp, reflectively1, reflectively2)


class Subtract(Shape):

    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

    def sdf(self, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        return np.where(sd1 < -sd2, -sd2, sd1)

    def rgb(self, x, y):
        return self.shape1.rgb(x, y)

    def reflectivity(self, x, y):
        return self.shape1.reflectivity(x, y)


class Circle(Shape):

    def __init__(self, x, y, r, red, green, blue, reflectivity0):
        self.x = x
        self.y = y
        self.r = r
        self.red = red
        self.green = green
        self.blue = blue
        self.reflectivity0 = reflectivity0

    def sdf(self, x, y):
        return np.hypot(x - self.x, y - self.y) - self.r

    def rgb(self, x, y):
        return self.red, self.green, self.blue

    def reflectivity(self, x, y):
        return self.reflectivity0


class Plane(Shape):

    def __init__(self, x, y, nx, ny, r, g, b, reflectivity0):
        self.x = x
        self.y = y
        self.nx = nx
        self.ny = ny
        self.r = r
        self.g = g
        self.b = b
        self.reflectivity0 = reflectivity0

    def sdf(self, x, y):
        return (x - self.x) * self.nx + (y - self.y) * self.ny

    def rgb(self, x, y):
        return self.r, self.g, self.b

    def reflectivity(self, x, y):
        return self.reflectivity0


class Capsule(Shape):

    def __init__(self, ax, ay, bx, by, r, red, green, blue, reflectivity0):
        self.ax = ax
        self.ay = ay
        self.bx = bx
        self.by = by
        self.r = r
        self.red = red
        self.green = green
        self.blue = blue
        self.reflectivity0 = reflectivity0

    def sdf(self, x, y):
        return segment(x, y, self.ax, self.ay, self.bx, self.by) - self.r

    def rgb(self, x, y):
        return self.red, self.green, self.blue

    def reflectivity(self, x, y):
        return self.reflectivity0


class Rectangle(Shape):

    def __init__(self, cx, cy, theta, sx, sy, r, g, b, reflectivity0):
        self.cx = cx
        self.cy = cy
        self.theta = theta
        self.sx = sx
        self.sy = sy
        self.r = r
        self.g = g
        self.b = b
        self.reflectivity0 = reflectivity0

    def sdf(self, x, y):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        cx = x - self.cx
        cy = y - self.cy
        dx = np.abs(cx * c + cy * s) - self.sx
        dy = np.abs(cy * c - cx * s) - self.sy
        ax = np.maximum(dx, 0)
        ay = np.maximum(dy, 0)
        return np.minimum(np.maximum(dx, dy), 0) + np.hypot(ax, ay)

    def rgb(self, x, y):
        return self.r, self.g, self.b

    def reflectivity(self, x, y):
        return self.reflectivity0


class CornerRectangle(Corner):

    def __init__(self, cx, cy, theta, sx, sy, r, red, green, blue, reflectivity0):
        rectangle = Rectangle(cx, cy, theta, sx, sy, red, green, blue, reflectivity0)
        super(CornerRectangle, self).__init__(rectangle, r)


class Triangle(Shape):

    def __init__(self, ax, ay, bx, by, cx, cy, r, g, b, reflectivity0):
        self.ax = ax
        self.ay = ay
        self.bx = bx
        self.by = by
        self.cx = cx
        self.cy = cy
        self.r = r
        self.g = g
        self.b = b
        self.reflectivity0 = reflectivity0

    def sdf(self, x, y):
        d0 = ((self.bx - self.ax) * (y - self.ay) > (self.by - self.ay) * (x - self.ax)) * \
             ((self.cx - self.bx) * (y - self.by) > (self.cy - self.by) * (x - self.bx)) * \
             ((self.ax - self.cx) * (y - self.cy) > (self.ay - self.cy) * (x - self.cx))
        d1 = minimum(
            segment(x, y, self.ax, self.ay, self.bx, self.by),
            segment(x, y, self.bx, self.by, self.cx, self.cy),
            segment(x, y, self.cx, self.cy, self.ax, self.ay),
        )
        return np.where(d0, -d1, d1)

    def rgb(self, x, y):
        return self.r, self.g, self.b

    def reflectivity(self, x, y):
        return self.reflectivity0


class CornerTriangle(Corner):

    def __init__(self, ax, ay, bx, by, cx, cy, r, red, green, blue, reflectivity0):
        triangle = Triangle(ax, ay, bx, by, cx, cy, red, green, blue, reflectivity0)
        super(CornerTriangle, self).__init__(triangle, r)


# circle = Circle(100, 200, 50, 255 * 2, 255 * 2, 255 * 2, 0)
# box1 = Rectangle(400, 250, -2 * np.pi / 16, 50, 50, 0, 0, 0, 0.9)
# box2 = Rectangle(250, 400, -2 * np.pi / 16, 50, 50, 0, 0, 0, 0)
# shape = Union(Union(circle, box1), box2)

# circle1 = Circle(150, 250, 50, 255 * 2, 255 * 2, 255 * 2, 0)
# circle2 = Circle(350, 250, 50, 0, 0, 0, 0.9)
# shape = Union(circle1, circle2)

# step = 64
# epsilon = 1e-6
#
# W = 500
# H = 500
# S = 64

# circle = Circle(40, 40, 20, 255 * 2, 255 * 2, 255 * 2, 0)
# plane = Plane(100, 100, -1, 0, 0, 0, 0, 0.9)
# shape = Union(circle, plane)

circle = Circle(40, 40, 20, 255 * 2, 255 * 2, 255 * 2, 0)
box1 = Rectangle(150, 100, -2 * np.pi / 16, 20, 20, 0, 0, 0, 0)
box2 = Rectangle(100, 150, -2 * np.pi / 16, 20, 20, 0, 0, 0, 0)
shape = Union(Union(circle, box1), box2)

step = 300
epsilon = 1e-6

W = 200
H = 200
S = 64

x = np.tile(np.arange(0, W), (H, 1)).T
y = np.tile(np.arange(0, H), (W, 1))
x = np.repeat(x[:, :, np.newaxis], S, axis=2)
y = np.repeat(y[:, :, np.newaxis], S, axis=2)

theta = 2 * np.pi / S * (np.tile(np.arange(0, S), (W, H, 1)) + np.random.random((W, H, S)))
ix = np.cos(theta)
iy = np.sin(theta)
# theta = np.ones((W, H, S)) * np.pi / 4
# ix = np.cos(theta)
# iy = -np.sin(theta)


stop = np.zeros((W, H, S), dtype=np.bool)

r = np.zeros((W, H, S))
g = np.zeros((W, H, S))
b = np.zeros((W, H, S))
reflectivity = np.ones((W, H, S))

"""
sd = shape.sdf(x, y)
if not stop
    if sd < epsilon
        rgb = rgb + shape.rgb(x, y) * reflectivity
        reflectivity = shape.reflectivity(x, y)
        if reflectivity == 0
            stop = True
        else
            nx, ny = gradient(shape, x, y, epsilon)
            ix, iy = reflect(x, y, ix, iy)
            x = x + nx * epsilon
            y = y + nx * epsilon
    else
        x = x + ix * sd
        y = y + ix * sd
"""

for i in range(step):
    print(i)
    sd = shape.sdf(x, y)

    # block
    t = (~stop) * (sd >= epsilon)
    x = np.where(t, x + ix * sd, x)
    y = np.where(t, y + iy * sd, y)

    # block
    t = (~stop) * (sd < epsilon)
    r0, g0, b0 = shape.rgb(x, y)
    r += np.where(t, r0, 0) * reflectivity
    g += np.where(t, g0, 0) * reflectivity
    b += np.where(t, b0, 0) * reflectivity
    reflectivity = np.where(t, shape.reflectivity(x, y), reflectivity)

    # block
    t = (~stop) * (sd < epsilon) * (reflectivity != 0)
    nx, ny = gradient(shape, x, y, 1e-6)
    ix, iy = np.where(t, reflect(ix, iy, nx, ny), [ix, iy])
    x = np.where(t, x + 1e-4 * nx, x)
    y = np.where(t, y + 1e-4 * ny, y)

    # block
    t = (~stop) * (sd < epsilon) * (reflectivity == 0)
    stop = np.where(t, True, stop)


r = 1 / S * np.sum(r, axis=2)
g = 1 / S * np.sum(g, axis=2)
b = 1 / S * np.sum(b, axis=2)

image = np.stack([r, g, b], axis=2)
image = np.minimum(image, 255)
image = image.astype(np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("image", image)
cv2.waitKey(0)
