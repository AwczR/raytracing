import numpy as np
from PIL import Image
import MATH_based_on_numpy as mymath
import math

BACKGROUND_COLOR = np.array([0.1, 0.1, 0.1])
MAX_DEPTH = 5
WIDTH = 720
HEIGHT = 720
RED=np.array([1,0,0])
BLUE=np.array([0,0,1])
GREEN=np.array([0,1,0])

def normalize(x):
    return x / np.linalg.norm(x)

class Camera():
    def __init__(self, position, width, height, lens_length):
        self.position = position
        self.width = width
        self.height = height
        self.lens_length = lens_length

class Sphere():
    def __init__(self, position, radius, color, texture):
        self.position = position
        self.radius = radius
        self.color = color
        self.texture = texture

class Light():
    def __init__(self, position, color):
        self.position = position
        self.color = color

class Ground():
    def __init__(self,point0,point1,point2,point3,type):
        #point0 defines the direction;point1-3 defines the ground
        self.discription=np.array([point0,point1,point2,point3])
        self.type=type

def touchground(O,D,ground,Light):
    # input O,D,'ground'
    # output a color 
    if D[1]>0:
        return BACKGROUND_COLOR
    normalized_vector=mymath.normalized_vector(ground.discription[0],ground.discription[1],ground.discription[2],ground.discription[3])
    touch_point=mymath.compute_ray_plane_intersection(O,D,ground.discription[1],ground.discription[2],ground.discription[3])
    if ground.type=="black_and_white":
        touch_point_x_rounded = round(touch_point[0])
        touch_point_z_rounded = round(touch_point[2])
        if round(touch_point_x_rounded/10)%2==0:
            if round(touch_point_z_rounded/10)%2==0:
                return np.array([0,0,0])
            return np.array([1,1,1])
        else:
            if round(touch_point_z_rounded/10)%2==0:
                return np.array([1,1,1])
            return np.array([0,0,0])
    if ground.type=="red_and_blue":
        touch_point_x_rounded = round(touch_point[0])
        touch_point_z_rounded = round(touch_point[2])
        if round(touch_point_x_rounded/10)%2==0:
            if round(touch_point_z_rounded/10)%2==0:
                return RED
            return BLUE
        else:
            if round(touch_point_z_rounded/10)%2==0:
                return BLUE
            return RED
        
def intersect_sphere(O, D, sphere):
    A = np.dot(D, D)
    OS = O - sphere.position
    B = 2 * np.dot(D, OS)
    C = np.dot(OS, OS) - sphere.radius * sphere.radius
    disc = B * B - 4 * A * C
    if disc > 0:
        dist_sqrt = np.sqrt(disc)
        q = -0.5 * (B + np.sign(B) * dist_sqrt)
        t0 = q / A
        t1 = C / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def trace_ray(O, D, balls, lights, depth,ground):
    if depth >= MAX_DEPTH:
        
        return touchground(O,D,ground,lights)

    closest_t = np.inf
    closest_sphere = None

    # DRX will update some code for Triangle Painting ,comming soon(maybe xD)!

    # Find the closest sphere (if any)
    for obj in balls:
        t = intersect_sphere(O, D, obj)
        if t < closest_t:
            closest_t = t
            closest_sphere = obj

    if closest_sphere is None:
        # Ray did not hit any object
        color = touchground(O,D,ground,lights)
        return color

    # Compute intersection point
    P = O + closest_t * D
    N = normalize(P - closest_sphere.position) # Surface normal at intersection point

    color = np.zeros(3)
    for light in lights:
        to_light = normalize(light.position - P)
        color += compute_diffuse_reflection(N, to_light, light.color, closest_sphere.color)

    # Assuming a "simple" texture means we have a basic reflective material
    # Reflection vector
    if closest_sphere.texture=='simple':
        R = D - 2 * np.dot(D, N) * N
        reflected_color = trace_ray(P + R * 1e-4, R, balls, lights, depth + 1,ground)
        color = color * 0.8 + reflected_color * 0.2
        return np.clip(color, 0, 1)
    if closest_sphere.texture=='mirror':
        R = D - 2 * np.dot(D, N) * N
        reflected_color = trace_ray(P + R * 1e-4, R, balls, lights, depth + 1,ground)
        color=reflected_color*0.9+color*0.1
        return np.clip(color,0,1)

def compute_diffuse_reflection(N, L, light_color, sphere_color):
    cos_theta = np.dot(N, L)
    if cos_theta > 0:
        return light_color * sphere_color * cos_theta
    return np.zeros(3)

def run():
    img = Image.new('RGB', (WIDTH, HEIGHT))
    
    balls = [
        Sphere(np.array([-6, 0, 30]), 3.0, np.array([1, 1, 1]), 'simple'),
        Sphere(np.array([ 6, 0, 30]), 3.0, np.array([1, 1, 1]), 'simple'),
        Sphere(np.array([ 0, -4, 30]),5.0,np.array([1,1,1]),'mirror')
        
    ]
    lights = [
        Light(np.array([0, 100, -10]), np.array([1, 1, 1]))
    ]
    ground=Ground(np.array([0,0,0]),np.array([-10,-9,10]),np.array([-10,-9,-10]),np.array([10,-9,10]),'black_and_white')
    aspect_ratio = float(WIDTH) / HEIGHT
    for j in range(HEIGHT):
        for i in range(WIDTH):
            # Convert pixel coordinate to world coordinate
            x = (2 * (i + 0.5) / WIDTH - 1) * aspect_ratio
            y = (1 - 2 * (j + 0.5) / HEIGHT)
            D = normalize(np.array([x, y, 1]))
            color = trace_ray(np.array([0., 0., 0.]), D, balls, lights, 0,ground)
            img.putpixel((i, j), tuple((color*255).astype(np.uint8))) # type: ignore

    img.save('test15.png')
    img.show()

if __name__ == "__main__":
    run()
