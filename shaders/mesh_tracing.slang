// Adapted from balboa. Copyright (c) 2026 Tzu-Mao Li. See THIRD_PARTY_NOTICES.md.
import geometry;

bool intersect_mesh(Ray ray, TriangleMesh mesh,
                      StructuredBuffer<float3> vertices,
                      StructuredBuffer<float3> vertex_normals,
                      StructuredBuffer<uint3> triangles,
                      StructuredBuffer<BVHNode> bvh_nodes, bool use_bvh,
                      float t_max, out float closest_t, out float3 normal) {
    closest_t = t_max;
    normal = float3(0.0);
    bool found = false;
    int node_index = use_bvh ? int(mesh.bvh_root_index) : 0;
    int node_end = use_bvh ? bvh_nodes[node_index].next_index : 1;
    while (node_index < node_end) {
        uint begin = mesh.triangle_begin;
        uint end = mesh.triangle_end;
        if (use_bvh) {
            BVHNode node = bvh_nodes[node_index];
            if (!intersect_aabb(node.bbox_min, node.bbox_max, ray.origin,
                                ray.direction, EPSILON, closest_t)) {
                node_index = node.next_index;
                continue;
            }
            if (!node.is_leaf) {
                node_index++;
                continue;
            }
            begin = node.triangle_begin;
            end = node.triangle_end;
        }
        for (uint i = begin; i < end; i++) {
            uint3 tri = triangles[i] + mesh.vertex_offset;
            float3 barycentrics;
            float t;
            if (intersect_tri(vertices[tri.x], vertices[tri.y], vertices[tri.z],
                               ray.origin, ray.direction, EPSILON, closest_t,
                               barycentrics, t)) {
                closest_t = t;
                normal = barycentrics.x * vertex_normals[tri.x]
                       + barycentrics.y * vertex_normals[tri.y]
                       + barycentrics.z * vertex_normals[tri.z];
                if (dot(normal, normal) == 0) {
                    normal = cross(vertices[tri.y] - vertices[tri.x],
                                   vertices[tri.z] - vertices[tri.x]);
                }
                found = true;
            }
        }
        node_index++;
    }
    return found;
}

bool closest_scene_hit(Ray ray, float t_max,
                        StructuredBuffer<Shape> shapes, int num_shapes,
                        StructuredBuffer<Sphere> spheres,
                        StructuredBuffer<TriangleMesh> meshes,
                        StructuredBuffer<float3> vertices,
                        StructuredBuffer<float3> vertex_normals,
                        StructuredBuffer<uint3> triangles,
                        StructuredBuffer<BVHNode> bvh_nodes, bool use_bvh,
                        out Hit hit) {
    hit = Hit(t_max, float3(0.0), float3(0.0), -1);
    for (int i = 0; i < num_shapes; i++) {
        Shape shape = shapes[i];
        Ray local_ray = object_ray(ray, shape);
        float t;
        float3 normal;
        bool found;
        if (shape.type == ShapeType.Sphere) {
            found = intersect_sphere(local_ray, spheres[shape.index], EPSILON,
                                      hit.t, t, normal);
        } else {
            found = intersect_mesh(local_ray, meshes[shape.index], vertices,
                                    vertex_normals, triangles, bvh_nodes,
                                    use_bvh, hit.t, t, normal);
        }
        if (found) {
            hit.t = t;
            hit.position = ray.origin + t * ray.direction;
            hit.normal = world_normal(shape, normal);
            hit.shape_index = i;
        }
    }
    return hit.shape_index >= 0;
}
