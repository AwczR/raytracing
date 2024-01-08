import numpy as np

def normalized_vector(point0,point1,point2,point3):
    #这个函数输入：原点， 三个顶点；
    #这个函数输出：三个顶点构成平面的法向量（背离原点方向）（模长归为1）
    vector1=point2-point1
    vector2=point3-point2
    temvector=point0-point1
    normal_vector=np.cross(vector1,vector2)
    if np.dot(normal_vector, temvector)>0:
        normal_vector=-normal_vector
    magnitude = np.linalg.norm(normal_vector)
    normalized_vector=normal_vector/magnitude
    return normalized_vector

def cosine_similarity(vector1,vector2):
    #这个函数输入两个向量
    #这个函数输出两个向量的余弦相似度，(-1,1),接近1表示两个向量相似
    dot_product = np.dot(vector1,vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2) # type: ignore
    cosine_similarity = dot_product / magnitude_product
    return cosine_similarity

def compute_ray_plane_intersection(ray_origin, ray_direction, point1, point2, point3):
    #这个函数输入：射线原点，射线方向，三个点坐标
    #输出交点坐标
    plane_normal = np.cross(point2 - point1, point3 - point1)
    t = np.dot(point1 - ray_origin, plane_normal) / np.dot(ray_direction, plane_normal)
    intersection_point = ray_origin + t * ray_direction
    return intersection_point

def isintriangle(point0,point1,point2,point3):
    #函数输入point0，point1-3
    #函数输出：如果point0在point1-3构成的图形内，returnTrue，elseFalse
    #请确保输入的点在同一平面内！！此函数只对有效输入负责
    vector1=point1-point0
    vector2=point2-point0
    vector3=point3-point0
    if np.dot((vector1+vector2),vector3)>0:
        return False
    if np.dot((vector3+vector2),vector1)>0:
        return False
    if np.dot((vector1+vector3),vector2)>0:
        return False
    return True
import numpy as np

def reflect_vector(incoming_direction, plane_normal):
    #输入：入射方向向量，法向量
    #输出：反射方向向量
    incoming_direction_normalized = incoming_direction / np.linalg.norm(incoming_direction)
    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    reflected_direction = incoming_direction_normalized - 2 * np.dot(incoming_direction_normalized, plane_normal_normalized) * plane_normal_normalized
    return reflected_direction

def intersect_ray_sphere(ray_origin, ray_direction, sphere_center, sphere_radius):
    #请设计一个函数，输入是一个射线的起点和方向向量，和一个球体的半径和球心，
    #返回射线与球的第一个焦点（默认有解）（射线起点在球体内部的情况也包含在内）
    # 将输入转换为NumPy数组
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    sphere_center = np.array(sphere_center)

    # 计算射线起点到球心的向量
    oc = ray_origin - sphere_center

    # 计算射线方向的单位向量
    unit_direction = ray_direction / np.linalg.norm(ray_direction)

    # 计算射线方向向量与球心到射线起点的向量的点积
    a = np.dot(unit_direction, unit_direction)
    b = 2.0 * np.dot(oc, unit_direction)
    c = np.dot(oc, oc) - sphere_radius ** 2

    # 计算判别式
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # 判别式小于0，没有实数解，说明射线与球体不相交
        return None

    # 计算两个解的参数化距离
    t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

    # 选择第一个焦点
    t = min(t1, t2)

    # 计算焦点坐标
    intersection_point = ray_origin + t * unit_direction

    return intersection_point

def refract_ray(ray_direction, normal, refractive_index):
    # 将输入转换为NumPy数组
    ray_direction = np.array(ray_direction)
    normal = np.array(normal)

    # 计算射线方向的单位向量
    unit_direction = ray_direction / np.linalg.norm(ray_direction)

    # 计算入射角的余弦值
    cos_theta_i = -np.dot(unit_direction, normal)

    if cos_theta_i < 0:
        # 光线从介质外部射入介质内部
        refractive_ratio = 1.0 / refractive_index
        cos_theta_i = -cos_theta_i
    else:
        # 光线从介质内部射出介质外部
        refractive_ratio = refractive_index
        normal = -normal

    cos_theta_r = np.sqrt(1 - (refractive_ratio ** 2) * (1 - cos_theta_i ** 2))

    # 计算折射后的方向向量
    refracted_direction = refractive_ratio * unit_direction + (refractive_ratio * cos_theta_i - cos_theta_r) * normal

    return refracted_direction

