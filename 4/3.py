from abc import ABC, abstractmethod

import cv2
import numpy as np
import tensorflow as tf


def hypot(x, y):
    return tf.sqrt(tf.pow(x, 2) + tf.pow(y, 2))


def segment(x, y, ax, ay, bx, by):
    vx = x - ax
    vy = y - ay
    ux = bx - ax
    uy = by - ay

    t = (vx * ux + vy * uy) / (ux * ux + uy * uy)
    t = tf.maximum(tf.minimum(t, 1), 0)
    dx = vx - ux * t
    dy = vy - uy * t
    return hypot(dx, dy)


def minimum(a, b, c):
    return tf.minimum(tf.minimum(a, b), c)


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
        self.r = tf.cast(r, dtype=tf.float64)

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
        return tf.where(self.tmp, sd2, sd1)

    def rgb(self, x, y):
        r1, g1, b1 = self.shape1.rgb(x, y)
        r2, g2, b2 = self.shape2.rgb(x, y)
        r = tf.where(self.tmp, r2, r1)
        g = tf.where(self.tmp, g2, g1)
        b = tf.where(self.tmp, b2, b1)
        return r, g, b

    def reflectivity(self, x, y):
        reflectively1 = self.shape1.reflectivity(x, y)
        reflectively2 = self.shape2.reflectivity(x, y)
        return tf.where(self.tmp, reflectively2, reflectively1)


class Union(Shape):

    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2
        self.tmp = None

    def sdf(self, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        self.tmp = sd1 < sd2
        return tf.where(self.tmp, sd1, sd2)

    def rgb(self, x, y):
        r1, g1, b1 = self.shape1.rgb(x, y)
        r2, g2, b2 = self.shape2.rgb(x, y)
        r = tf.where(self.tmp, r1, r2)
        g = tf.where(self.tmp, g1, g2)
        b = tf.where(self.tmp, b1, b2)
        return r, g, b

    def reflectivity(self, x, y):
        reflectively1 = self.shape1.reflectivity(x, y)
        reflectively2 = self.shape2.reflectivity(x, y)
        return tf.where(self.tmp, reflectively1, reflectively2)


class Subtract(Shape):

    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

    def sdf(self, x, y):
        sd1 = self.shape1.sdf(x, y)
        sd2 = self.shape2.sdf(x, y)
        return tf.where(sd1 < -sd2, -sd2, sd1)

    def rgb(self, x, y):
        return self.shape1.rgb(x, y)

    def reflectivity(self, x, y):
        return self.shape1.reflectivity(x, y)


class Supply(Shape):

    def __init__(self, shape):
        self.shape = shape

    def sdf(self, x, y):
        return -self.shape.sdf(x, y)

    def rgb(self, x, y):
        return self.shape.rgb(x, y)

    def reflectivity(self, x, y):
        return self.shape.reflectivity(x, y)


class Circle(Shape):

    def __init__(self, x, y, r, red, green, blue, reflectivity0):
        self.x = tf.cast(x, dtype=tf.float64)
        self.y = tf.cast(y, dtype=tf.float64)
        self.r = tf.cast(r, dtype=tf.float64)
        self.red = tf.cast(red, dtype=tf.float64)
        self.green = tf.cast(green, dtype=tf.float64)
        self.blue = tf.cast(blue, dtype=tf.float64)
        self.reflectivity0 = tf.cast(reflectivity0, dtype=tf.float64)

    def sdf(self, x, y):
        return hypot(x - self.x, y - self.y) - self.r

    def rgb(self, x, y):
        return self.red, self.green, self.blue

    def reflectivity(self, x, y):
        return self.reflectivity0


class Eclipse(Shape):

    def __init__(self, x0, y0, x1, y1, a, red, green, blue, reflectivity0):
        self.x0 = tf.cast(x0, dtype=tf.float64)
        self.y0 = tf.cast(y0, dtype=tf.float64)
        self.x1 = tf.cast(x1, dtype=tf.float64)
        self.y1 = tf.cast(y1, dtype=tf.float64)
        self.a = tf.cast(a, dtype=tf.float64)
        self.red = tf.cast(red, dtype=tf.float64)
        self.green = tf.cast(green, dtype=tf.float64)
        self.blue = tf.cast(blue, dtype=tf.float64)
        self.reflectivity0 = tf.cast(reflectivity0, dtype=tf.float64)

    def sdf(self, x, y):
        return hypot(x - self.x0, y - self.y0) / 2 + hypot(x - self.x1, y - self.y1) / 2 - self.a

    def rgb(self, x, y):
        return self.red, self.green, self.blue

    def reflectivity(self, x, y):
        return self.reflectivity0


class Hyperbola(Shape):

    def __init__(self, x0, y0, x1, y1, a, red, green, blue, reflectivity0):
        self.x0 = tf.cast(x0, dtype=tf.float64)
        self.y0 = tf.cast(y0, dtype=tf.float64)
        self.x1 = tf.cast(x1, dtype=tf.float64)
        self.y1 = tf.cast(y1, dtype=tf.float64)
        self.a = tf.cast(a, dtype=tf.float64)
        self.red = tf.cast(red, dtype=tf.float64)
        self.green = tf.cast(green, dtype=tf.float64)
        self.blue = tf.cast(blue, dtype=tf.float64)
        self.reflectivity0 = tf.cast(reflectivity0, dtype=tf.float64)

    def sdf(self, x, y):
        return hypot(x - self.x0, y - self.y0) / 2 - hypot(x - self.x1, y - self.y1) / 2 - self.a

    def rgb(self, x, y):
        return self.red, self.green, self.blue

    def reflectivity(self, x, y):
        return self.reflectivity0


class Plane(Shape):

    def __init__(self, x, y, nx, ny, r, g, b, reflectivity0):
        self.x = tf.cast(x, dtype=tf.float64)
        self.y = tf.cast(y, dtype=tf.float64)
        self.nx = tf.cast(nx, dtype=tf.float64)
        self.ny = tf.cast(ny, dtype=tf.float64)
        self.r = tf.cast(r, dtype=tf.float64)
        self.g = tf.cast(g, dtype=tf.float64)
        self.b = tf.cast(b, dtype=tf.float64)
        self.reflectivity0 = tf.cast(reflectivity0, dtype=tf.float64)

    def sdf(self, x, y):
        return (x - self.x) * self.nx + (y - self.y) * self.ny

    def rgb(self, x, y):
        return self.r, self.g, self.b

    def reflectivity(self, x, y):
        return self.reflectivity0


class Capsule(Shape):

    def __init__(self, ax, ay, bx, by, r, red, green, blue, reflectivity0):
        self.ax = tf.cast(ax, dtype=tf.float64)
        self.ay = tf.cast(ay, dtype=tf.float64)
        self.bx = tf.cast(bx, dtype=tf.float64)
        self.by = tf.cast(by, dtype=tf.float64)
        self.r = tf.cast(r, dtype=tf.float64)
        self.red = tf.cast(red, dtype=tf.float64)
        self.green = tf.cast(green, dtype=tf.float64)
        self.blue = tf.cast(blue, dtype=tf.float64)
        self.reflectivity0 = tf.cast(reflectivity0, dtype=tf.float64)

    def sdf(self, x, y):
        return segment(x, y, self.ax, self.ay, self.bx, self.by) - self.r

    def rgb(self, x, y):
        return self.red, self.green, self.blue

    def reflectivity(self, x, y):
        return self.reflectivity0


class Rectangle(Shape):

    def __init__(self, cx, cy, theta, sx, sy, r, g, b, reflectivity0):
        self.cx = tf.cast(cx, dtype=tf.float64)
        self.cy = tf.cast(cy, dtype=tf.float64)
        self.theta = tf.cast(theta, dtype=tf.float64)
        self.sx = tf.cast(sx, dtype=tf.float64)
        self.sy = tf.cast(sy, dtype=tf.float64)
        self.r = tf.cast(r, dtype=tf.float64)
        self.g = tf.cast(g, dtype=tf.float64)
        self.b = tf.cast(b, dtype=tf.float64)
        self.reflectivity0 = tf.cast(reflectivity0, dtype=tf.float64)

    def sdf(self, x, y):
        c = tf.cos(self.theta)
        s = tf.sin(self.theta)
        cx = x - self.cx
        cy = y - self.cy
        dx = tf.abs(cx * c + cy * s) - self.sx
        dy = tf.abs(cy * c - cx * s) - self.sy
        ax = tf.maximum(dx, 0)
        ay = tf.maximum(dy, 0)
        return tf.minimum(tf.maximum(dx, dy), 0) + hypot(ax, ay)

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
        self.ax = tf.cast(ax, dtype=tf.float64)
        self.ay = tf.cast(ay, dtype=tf.float64)
        self.bx = tf.cast(bx, dtype=tf.float64)
        self.by = tf.cast(by, dtype=tf.float64)
        self.cx = tf.cast(cx, dtype=tf.float64)
        self.cy = tf.cast(cy, dtype=tf.float64)
        self.r = tf.cast(r, dtype=tf.float64)
        self.g = tf.cast(g, dtype=tf.float64)
        self.b = tf.cast(b, dtype=tf.float64)
        self.reflectivity0 = tf.cast(reflectivity0, dtype=tf.float64)

    def sdf(self, x, y):
        d0 = ((self.bx - self.ax) * (y - self.ay) > (self.by - self.ay) * (x - self.ax)) & \
             ((self.cx - self.bx) * (y - self.by) > (self.cy - self.by) * (x - self.bx)) & \
             ((self.ax - self.cx) * (y - self.cy) > (self.ay - self.cy) * (x - self.cx))
        d1 = minimum(
            segment(x, y, self.ax, self.ay, self.bx, self.by),
            segment(x, y, self.bx, self.by, self.cx, self.cy),
            segment(x, y, self.cx, self.cy, self.ax, self.ay),
        )
        return tf.where(d0, -d1, d1)

    def rgb(self, x, y):
        return self.r, self.g, self.b

    def reflectivity(self, x, y):
        return self.reflectivity0


class CornerTriangle(Corner):

    def __init__(self, ax, ay, bx, by, cx, cy, r, red, green, blue, reflectivity0):
        triangle = Triangle(ax, ay, bx, by, cx, cy, red, green, blue, reflectivity0)
        super(CornerTriangle, self).__init__(triangle, r)


class Light2D:

    def __init__(self, shape, w, h, d, step):
        self.shape = shape
        self.w = w
        self.h = h
        self.d = d
        self.step = step

    def render(self):
        """
        sd = shape.sdf(x, y)
        if not stop
            if sd >= epsilon
                x = x + ix * sd
                y = y + ix * sd
            else
                rgb = rgb + shape.rgb(x, y) * reflectivity
                reflectivity = shape.reflectivity(x, y)
                if sd >= 0 and reflectivity != 0
                    nx, ny = gradient(shape, x, y, epsilon)
                    ix, iy = reflect(x, y, ix, iy)
                    x = x + nx * epsilon
                    y = y + nx * epsilon
                else
                    stop = True
        """
        x = tf.transpose(tf.reshape(tf.tile(tf.range(0, self.w, dtype=tf.float64), [self.h]), (self.h, self.w)))
        y = tf.transpose(x)
        x = tf.repeat(x[:, :, tf.newaxis], self.d, axis=2)
        y = tf.repeat(y[:, :, tf.newaxis], self.d, axis=2)

        theta = tf.reshape(tf.tile(tf.range(0, self.d, dtype=tf.float64), [self.w * self.h]), (self.w, self.h, self.d))
        theta = theta + tf.random.uniform((self.w, self.h, self.d), dtype=tf.float64)
        theta = 2 * np.pi / self.d * theta
        ix = tf.cos(theta)
        iy = tf.sin(theta)

        stop = tf.zeros((self.w, self.h, self.d), dtype=tf.bool)
        reflectivity = tf.ones((self.w, self.h, self.d), dtype=tf.float64)
        r = tf.zeros((self.w, self.h, self.d), dtype=tf.float64)
        g = tf.zeros((self.w, self.h, self.d), dtype=tf.float64)
        b = tf.zeros((self.w, self.h, self.d), dtype=tf.float64)

        epsilon = 1e-6
        for i in range(self.step):
            print(i)
            sd = self.shape.sdf(x, y)

            # with tf.GradientTape() as tape:
            #     tape.watch([x, y])
            #     sd = self.shape.sdf(x, y)
            # nx, ny = tape.gradient(sd, [x, y])

            # block
            t = (~stop) & (sd >= epsilon)
            x = tf.where(t, x + ix * sd, x)
            y = tf.where(t, y + iy * sd, y)

            # block
            t = (~stop) & (sd < epsilon)
            r0, g0, b0 = self.shape.rgb(x, y)
            r += tf.where(t, r0, 0) * reflectivity
            g += tf.where(t, g0, 0) * reflectivity
            b += tf.where(t, b0, 0) * reflectivity
            reflectivity = tf.where(t, self.shape.reflectivity(x, y), reflectivity)

            # block
            t = (~stop) & (sd < epsilon) & ((sd >= 0) & (reflectivity != 0))
            nx, ny = gradient(self.shape, x, y, epsilon)
            ix, iy = tf.where(t, reflect(ix, iy, nx, ny), [ix, iy])
            x = tf.where(t, x + 1e-4 * nx, x)
            y = tf.where(t, y + 1e-4 * ny, y)

            # block
            t = (~stop) & (sd < epsilon) & ((sd < 0) | (reflectivity == 0))
            stop = tf.where(t, True, stop)

        r = 1 / self.d * tf.reduce_sum(r, axis=2)
        g = 1 / self.d * tf.reduce_sum(g, axis=2)
        b = 1 / self.d * tf.reduce_sum(b, axis=2)

        image = tf.stack([r, g, b], axis=2)
        image = tf.minimum(image, 255)
        image = image.numpy()
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


# 1
circle = Circle(100, 200, 50, 255 * 2, 255 * 2, 255 * 2, 0)
box1 = Rectangle(400, 250, -2 * np.pi / 16, 50, 50, 0, 0, 0, 0.9)
box2 = Rectangle(250, 400, -2 * np.pi / 16, 50, 50, 0, 0, 0, 0.9)
shape = Union(Union(circle, box1), box2)
light2d = Light2D(shape, 500, 500, 64, 300)
image = light2d.render()
cv2.imwrite("1.jpg", image)

# 2
# triangle = Circle(80, 80, 50, 255 * 5, 255 * 5, 255 * 5, 0)
# box1 = Rectangle(250, 75, 0, 10, 75, 0, 0, 0, 0)
# box2 = Rectangle(250, 350, 0, 10, 150, 0, 0, 0, 0)
# box3 = Rectangle(400, 250, 0, 10, 250, 0, 0, 0, 1)
# shape = Union(Union(Union(triangle, box1), box2), box3)
# light2d = Light2D(shape, 500, 500, 64, 1000)
# image = light2d.render()
# cv2.imwrite("1.jpg", image)

# 3
# circle1 = Circle(100, 200, 50, 255 * 2, 255 * 2, 255 * 2, 0)
# plane = Plane(250, 0, -1, 0, 0, 0, 0, 0.9)
# circle2 = Circle(250, 250, 200, 0, 0, 0, 0.9)
# shape = Union(circle1, Subtract(plane, circle2))
# light2d = Light2D(shape, 500, 500, 64, 300)
# image = light2d.render()
# cv2.imwrite("1.jpg", image)

# 4
# circle = Circle(300, 200, 50, 255 * 2, 255 * 2, 255 * 2, 0)
# eclipse = Eclipse(250, 100, 250, 400, 200, 0, 0, 0, 1)
# shape = Union(circle, Supply(eclipse))
# light2d = Light2D(shape, 500, 500, 64, 300)
# image = light2d.render()
# cv2.imwrite("1.jpg", image)

# 5
# circle = Circle(150, 350, 50, 255 * 3, 255 * 3, 255 * 3, 0)
# hyperbola = Hyperbola(250, -150, 250, 150, 100, 0, 0, 0, 1)
# shape = Union(circle, hyperbola)
# light2d = Light2D(shape, 500, 500, 64, 300)
# image = light2d.render()
# cv2.imwrite("1.jpg", image)
