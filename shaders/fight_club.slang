// Ray-traced surfaces, lighting, and smoke for fight_club.
import geometry;
import mesh_tracing;
import volumes;

uniform Camera camera;
uniform int num_shapes;
uniform int num_lights;
uniform int num_floor_fires;
uniform float2 resolution;
StructuredBuffer<Shape> shapes;
StructuredBuffer<Sphere> spheres;
StructuredBuffer<TriangleMesh> meshes;
StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<uint3> triangles;
StructuredBuffer<BVHNode> bvh_nodes;
StructuredBuffer<PointLight> lights;
StructuredBuffer<uint> material_kinds;
StructuredBuffer<uint> material_seeds;
StructuredBuffer<float> window_intensities;
StructuredBuffer<FloorFire> floor_fires;
RWTexture2D<float4> output_image;

float hash21(float2 p) {
    return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float noise2(float2 p) {
    float2 cell = floor(p);
    float2 f = frac(p);
    f = f * f * (3 - 2 * f);
    return lerp(lerp(hash21(cell), hash21(cell + float2(1, 0)), f.x),
                lerp(hash21(cell + float2(0, 1)), hash21(cell + 1), f.x), f.y);
}

float fractal_noise(float2 p) {
    float value = 0;
    float weight = .5;
    for (int i = 0; i < 4; i++) {
        value += weight * noise2(p);
        p = float2(p.x * 1.61 - p.y * 1.17, p.x * 1.17 + p.y * 1.61) + 9.7;
        weight *= .5;
    }
    return value;
}

float3 sky_color(float3 direction) {
    float elevation = saturate(direction.y * 2.5 + .13);
    float3 color = lerp(float3(.016, .043, .060), float3(.005, .015, .027), elevation);
    float cloud = fractal_noise(float2(direction.x * 6, direction.y * 11) + float2(4, 13));
    color += float3(.009, .018, .024) * smoothstep(.25, .68, cloud);
    return color;
}

bool scene_hit(Ray ray, float t_max, out Hit hit) {
    return closest_scene_hit(ray, t_max, shapes, num_shapes, spheres, meshes,
                             vertices, vertex_normals, triangles, bvh_nodes, true, hit);
}

float3 building_color(Hit hit, Shape shape) {
    // Rest-space coordinates keep facade lights attached to moving sections.
    float3 p = xform_pt(shape.world_to_obj, hit.position);
    float3 n = normalize(xform_vec(transpose(shape.obj_to_world), hit.normal));
    float side_coordinate = abs(n.x) > .5 ? p.z : p.x;
    float2 grid = float2(side_coordinate / .95, p.y / 1.75);
    float2 cell = floor(grid);
    float2 f = frac(grid);
    float window = smoothstep(.16, .23, f.x) * (1 - smoothstep(.77, .84, f.x));
    window *= smoothstep(.30, .37, f.y) * (1 - smoothstep(.69, .76, f.y));
    window *= 1 - step(.5, abs(n.y));
    float seed = material_seeds[hit.shape_index] * 19.13;
    float room = hash21(cell + seed);
    float office = hash21(floor(cell / float2(4, 1)) + seed * 1.7);
    float occupied = step(.31, room) * step(.21, office);
    float3 lamp = lerp(float3(.38, .56, .62), float3(.60, .51, .35), step(.86, room));
    float intensity = lerp(.18, 1.0, hash21(cell + 56.7));
    float mullion = .8 + .2 * step(.065, abs(f.x - .5));
    float3 color = shape.Kd * (.15 + .15 * max(0, dot(n, normalize(float3(-1, 2, 1)))));
    color += window * occupied * lamp * intensity * mullion * window_intensities[hit.shape_index];
    float floor_band = 1 - smoothstep(.015, .045, abs(frac((p.y + 4) / 18) - .5));
    color += floor_band * float3(.012, .027, .032) * (1 - abs(n.y));
    // Dark, narrow service floors break up large stretches of repeated windows.
    color *= .83 + .17 * step(.10, frac(p.y / 24));
    for (int i = 0; i < num_floor_fires; i++) {
        FloorFire fire = floor_fires[i];
        if (fire.material_seed != material_seeds[hit.shape_index]
            || abs(p.y - fire.rest_y) > .86 || dot(n, fire.rest_normal) < .9) continue;
        float flicker = .72 + .28 * hash21(cell + fire.rest_y * 7.1);
        float3 fire_color = lerp(float3(1, .22, .018), float3(1, .53, .11), room);
        color += window * mullion * flicker * fire.glow * fire_color;
    }
    return color;
}

float3 floor_fire_lighting(Hit hit, float3 view_direction, bool include_specular) {
    uint kind = material_kinds[hit.shape_index];
    float3 normal = normalize(hit.normal);
    if (dot(normal, view_direction) > 0) normal = -normal;
    float3 diffuse = kind == 1 ? float3(0) : shapes[hit.shape_index].Kd / 3.14159265;
    float3 color = float3(0);
    for (int i = 0; i < num_floor_fires; i++) {
        FloorFire fire = floor_fires[i];
        // Integrate the thin rectangle with eight stratified area samples.
        float open_area = 4 * length(cross(fire.half_u, fire.half_v)) * .56;
        for (int sample_index = 0; sample_index < 8; sample_index++) {
            float u = (sample_index / 2 + .5) * .5 - 1;
            float v = sample_index % 2 == 0 ? -.5 : .5;
            float3 position = fire.center + u * fire.half_u + v * fire.half_v;
            float3 to_light = position - hit.position;
            float distance_squared = max(dot(to_light, to_light), .01);
            float distance = sqrt(distance_squared);
            float3 direction = to_light / distance;
            float receive_cosine = max(0, dot(normal, direction));
            float emit_cosine = max(0, dot(fire.normal, -direction));
            if (receive_cosine * emit_cosine <= 0) continue;
            float3 weight = fire.radiance * (open_area * .125 * emit_cosine / distance_squared);
            if (max(weight.x, max(weight.y, weight.z)) < .00001) continue;
            Hit blocker;
            Ray shadow_ray = Ray(hit.position + normal * .002, direction);
            bool blocked = scene_hit(shadow_ray, distance - .004, blocker);
            if (blocked) continue;
            weight *= smoke_transmittance(shadow_ray, distance, 12);
            color += diffuse * receive_cosine * weight;
            if (include_specular) {
                float exponent = kind == 1 ? 512 : (kind == 2 ? 65 : 24);
                float strength = kind == 1 ? .00005 : (kind == 2 ? .025 : .003);
                float3 halfway = normalize(direction - view_direction);
                color += strength * pow(max(0, dot(normal, halfway)), exponent) * weight;
            }
        }
    }
    return color;
}

float3 surface_color(Hit hit, float3 view_direction, bool shadows) {
    Shape shape = shapes[hit.shape_index];
    uint kind = material_kinds[hit.shape_index];
    float3 normal = normalize(hit.normal);
    if (dot(normal, view_direction) > 0) normal = -normal;
    float3 color;
    if (kind == 1) {
        color = building_color(hit, shape);
    } else if (kind == 3) {
        color = shape.Kd;
    } else {
        float ambient = .040 + .065 * saturate(normal.y * .5 + .5);
        color = shape.Kd * float3(.62, .85, 1.0) * ambient;
    }
    if (kind != 3) {
        for (int i = 0; i < num_lights; i++) {
            if (kind == 1) continue;
            float3 to_light = lights[i].position - hit.position;
            float distance_squared = dot(to_light, to_light);
            float distance = sqrt(distance_squared);
            float3 light_direction = to_light / max(distance, .001);
            float ndotl = max(0, dot(normal, light_direction));
            if (ndotl <= 0) continue;
            Hit blocker;
            Ray shadow_ray = Ray(hit.position + normal * .002, light_direction);
            bool blocked = shadows && scene_hit(shadow_ray, distance - .004, blocker);
            if (!blocked) {
                // Glazed facades have low diffuse reflectance and a tighter highlight.
                float3 diffuse_color = shape.Kd;
                float exponent = kind == 2 ? 65 : 24;
                float specular_strength = kind == 2 ? .025 : .003;
                if (kind == 1) {
                    diffuse_color = float3(0);
                    exponent = 512;
                    specular_strength = .00005;
                }
                color += point_light_contribution(diffuse_color, normal, light_direction,
                                                   lights[i], distance_squared);
                float3 halfway = normalize(light_direction - view_direction);
                float specular = pow(max(0, dot(normal, halfway)), exponent);
                color += specular_strength * specular * lights[i].intensity / max(distance_squared, .01);
            }
        }
        color += floor_fire_lighting(hit, view_direction, true);
        if (kind != 1) {
            float rim = pow(1 - saturate(dot(normal, -view_direction)), 3);
            color += shape.Kd * float3(.10, .18, .22) * rim;
        }
    }
    if (hit.position.z < -.5) {
        float haze = 1 - exp(-max(0, -hit.position.z - 20) * .0022);
        color = lerp(color, float3(.014, .035, .047), haze);
    }
    return color;
}

float3 room_indirect(Hit hit, float3 view_direction) {
    uint kind = material_kinds[hit.shape_index];
    if (num_floor_fires == 0 || kind == 1 || kind == 3 || hit.position.z < -.35) return float3(0);
    float3 normal = normalize(hit.normal);
    if (dot(normal, view_direction) > 0) normal = -normal;
    float3 helper = abs(normal.y) < .9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(helper, normal));
    float3 bitangent = cross(normal, tangent);
    float3 incoming = float3(0);
    // Four cosine-weighted rays transfer ceiling and wall illumination into the room.
    for (int i = 0; i < 4; i++) {
        float radius = sqrt((i + .5) / 4);
        float angle = i * 2.39996323 + .4;
        float3 direction = radius * (cos(angle) * tangent + sin(angle) * bitangent)
                         + sqrt(1 - radius * radius) * normal;
        Ray bounce = Ray(hit.position + normal * .003, direction);
        Hit receiver;
        if (!scene_hit(bounce, 14, receiver) || receiver.position.z < -.35
            || material_kinds[receiver.shape_index] == 3) continue;
        incoming += floor_fire_lighting(receiver, direction, false);
    }
    return shapes[hit.shape_index].Kd * incoming * .25;
}

float3 reflected_color(Ray ray) {
    Hit reflected_hit;
    bool found = scene_hit(ray, RAY_MAX, reflected_hit);
    float3 color = found ? surface_color(reflected_hit, ray.direction, false) : sky_color(ray.direction);
    return composite_explosions(ray, found ? reflected_hit.t : RAY_MAX, color);
}

float3 shade_hit(Ray ray, Hit hit) {
    float3 color = surface_color(hit, ray.direction, true);
    color += room_indirect(hit, ray.direction);
    if (material_kinds[hit.shape_index] == 2) {
        float2 p = hit.position.xz;
        float2 distortion = float2(noise2(p * 5.0), noise2(p * 5.0 + 33)) - .5;
        float3 normal = normalize(float3(distortion.x * .012, 1, distortion.y * .012));
        Ray reflection = Ray(hit.position + float3(0, .003, 0), reflect(ray.direction, normal));
        float fresnel = .065 + .22 * pow(1 - saturate(dot(normal, -ray.direction)), 5);
        color = color * (1 - fresnel) + reflected_color(reflection) * fresnel * float3(.65, .82, .87);
        float concrete = .96 + .04 * fractal_noise(p * 19);
        color *= concrete;
    }
    return max(color, float3(0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID) {
    if (any(tid >= uint2(resolution))) return;
    Ray ray = camera_ray(tid, uint(resolution.x), uint(resolution.y), camera);
    Hit hit;
    bool found = scene_hit(ray, RAY_MAX, hit);
    float3 color = found ? shade_hit(ray, hit) : sky_color(ray.direction);
    color = composite_explosions(ray, found ? hit.t : RAY_MAX, color);
    output_image[tid] = float4(color, 1);
}
