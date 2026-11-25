// Adapted from balboa. Copyright (c) 2026 Tzu-Mao Li. See THIRD_PARTY_NOTICES.md.
// Common scene definitions
enum ShapeType {
    Sphere,
    TriangleMesh
};

struct Camera {
    float4x4 cam_to_world;
    float fov_x;
};

struct Sphere {
    float3 center;
    float radius;
};

struct TriangleMesh {
    uint vertex_offset;
    uint triangle_begin;
    uint triangle_end;
    uint bvh_root_index;
};

struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;

    // Skip here on a ray miss. A negative index ends traversal.
    int next_index;

    uint triangle_begin;
    uint triangle_end;

    bool is_leaf;
};

struct Shape {
    ShapeType type;
    // Index the shape into the type-specific arrays
    // (spheres and triangle_mesh)
    int index;

    float3 Kd;
    float3 Kr;

    // The following are always affine transforms.
    // Maps world-space positions to the shape's object space.
    float4x4 world_to_obj;
    // Inverse of above.
    float4x4 obj_to_world;
};

struct PointLight {
    float3 position;
    float3 intensity;
};

struct FloorFire {
    float3 center;
    float3 half_u;
    float3 half_v;
    float3 normal;
    float3 rest_normal;
    float3 radiance;
    float glow;
    float rest_y;
    uint material_seed;
};

static const float EPSILON = 1e-3;

// Return true when t contains the real roots.
bool solve_quadratic(float a, float b, float c,
                     out float2 t) {
    // From https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/math.h#L613
    if (a == 0) {
        if (b == 0) {
            t = float2(0.0, 0.0);
            return false;
        }
        t = float2(-c / b, -c / b);
        return true;
    }

    float discrim = b * b - 4 * a * c;
    if (discrim < 0) {
        t = float2(0, 0);
        return false;
    }
    float root_discrim = sqrt(discrim);
    if (root_discrim == 0) {
        t = float2(-b / (2 * a), -b / (2 * a));
        return true;
    }

    float q;
    if (b < 0) {
        q = -0.5f * (b - root_discrim);
    } else {
        q = -0.5f * (b + root_discrim);
    }
    t.x = q / a;
    t.y = c / q;
    return true;
}

float3 xform_pt(float4x4 xform, float3 pt) {
    return mul(xform, float4(pt, 1.0)).xyz;
}

float3 xform_vec(float4x4 xform, float3 vec) {
    return mul(xform, float4(vec, 0.0)).xyz;
}

bool intersect_tri(float3 p0, float3 p1, float3 p2,
                   float3 ray_org,
                   float3 ray_dir,
                   float t_min,
                   float t_max,
                   out float3 b,
                   out float t) {
    // return the barycentric coordinates b
    // and intersection time t
    b = float3(0.0, 0.0, 0.0);
    t = t_max;

    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 s1 = cross(ray_dir, e2);
    float divisor = dot(s1, e1);
    if (divisor == 0) {
        return false;
    }

    float inv_divisor = 1 / divisor;
    float3 s = ray_org - p0;
    float u = dot(s, s1) * inv_divisor;
    float3 s2 = cross(s, e1);
    float v = dot(ray_dir, s2) * inv_divisor;
    t = dot(e2, s2) * inv_divisor;

    if (t > t_min && t < t_max && u >= 0 && v >= 0 && u + v <= 1) {
        b.x = 1 - u - v;
        b.y = u;
        b.z = v;
        return true;
    }

    t = t_max;
    return false;
}

bool intersect_aabb(float3 p_min,
                    float3 p_max,
                    float3 ray_org,
                    float3 ray_dir,
                    float t_min,
                    float t_max) {
    // Return true if intersect
    // https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies/rayintersectionwithanaabb
    for (int i = 0; i < 3; i++) {
        if (ray_dir[i] == 0) {
            if (ray_org[i] < p_min[i] || ray_org[i] > p_max[i]) {
                return false;
            }
            continue;
        }
        // This code was designed to work even when ray_dir[i] == 0
        // See https://pbr-book.org/4ed/Shapes/Basic_Shape_Interface
        float inv_dir = float(1) / ray_dir[i];
        float t0 = (p_min[i] - ray_org[i]) * inv_dir;
        float t1 = (p_max[i] - ray_org[i]) * inv_dir;
        if (inv_dir < 0) {
            // swap
            float _tmp = t1;
            t1 = t0;
            t0 = _tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max < t_min) {
            return false;
        }
    }
    return true;
}

static const float RAY_MAX = 1e30;

struct Ray {
    float3 origin;
    float3 direction;
};

struct Hit {
    float t;
    float3 position;
    float3 normal;
    int shape_index;
};

Ray camera_ray(uint2 pixel, uint width, uint height, Camera camera) {
    float2 uv = (float2(pixel) + 0.5) / float2(width, height);
    float half_width = tan(radians(camera.fov_x) * 0.5);
    float half_height = half_width * float(height) / float(width);
    float3 direction = float3((2 * uv.x - 1) * half_width,
                              (1 - 2 * uv.y) * half_height, -1);
    Ray ray;
    ray.origin = xform_pt(camera.cam_to_world, float3(0.0));
    ray.direction = normalize(xform_vec(camera.cam_to_world, direction));
    return ray;
}

Ray object_ray(Ray ray, Shape shape) {
    Ray result;
    result.origin = xform_pt(shape.world_to_obj, ray.origin);
    // Keeping this direction unnormalized preserves t across objects.
    result.direction = xform_vec(shape.world_to_obj, ray.direction);
    return result;
}

bool intersect_sphere(Ray ray, Sphere sphere, float t_min, float t_max,
                      out float t, out float3 normal) {
    float3 offset = ray.origin - sphere.center;
    float2 roots;
    t = t_max;
    normal = float3(0.0);
    if (!solve_quadratic(dot(ray.direction, ray.direction),
                         2 * dot(offset, ray.direction),
                         dot(offset, offset) - sphere.radius * sphere.radius,
                         roots)) {
        return false;
    }
    float near_t = min(roots.x, roots.y);
    float far_t = max(roots.x, roots.y);
    t = near_t > t_min ? near_t : far_t;
    if (t <= t_min || t >= t_max) {
        return false;
    }
    normal = normalize(ray.origin + t * ray.direction - sphere.center);
    return true;
}

float3 world_normal(Shape shape, float3 normal) {
    return normalize(xform_vec(transpose(shape.world_to_obj), normal));
}

float3 point_light_contribution(float3 Kd, float3 normal,
                                 float3 light_direction,
                                 PointLight light, float distance_squared) {
    return Kd * light.intensity * max(dot(normal, light_direction), 0.0)
           / distance_squared;
}
