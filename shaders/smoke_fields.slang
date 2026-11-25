// Cache three-dimensional density and single-scattered illumination each frame.
import geometry;
import mesh_tracing;
import volumes;

uniform int num_shapes;
uniform int num_floor_fires;
StructuredBuffer<Shape> shapes;
StructuredBuffer<Sphere> spheres;
StructuredBuffer<TriangleMesh> meshes;
StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<uint3> triangles;
StructuredBuffer<BVHNode> bvh_nodes;
StructuredBuffer<FloorFire> floor_fires;
uniform Camera camera;
uniform float simulation_time;
uniform float smoke_detail;
uniform float3 smoke_wind;
uniform float smoke_lift;
Texture3D<float2> transported_smoke;
Texture3D<float4> smoke_direct_grid;
RWTexture3D<float2> density_output;
RWTexture3D<float4> lighting_output;

bool light_blocked(Ray ray, float distance) {
    Hit hit;
    return closest_scene_hit(ray, distance, shapes, num_shapes, spheres, meshes,
                             vertices, vertex_normals, triangles, bvh_nodes, true, hit);
}

float3 voxel_position(uint3 tid, int index, int resolution) {
    uint3 local = uint3(tid.xy, tid.z % resolution);
    float3 uv = (float3(local) + .5) / resolution;
    return explosions[index].center + (uv * 2 - 1) * explosions[index].half_extent;
}

float3 scattering_albedo(float dust_fraction) {
    return smoke_albedo * lerp(float3(.55, .54, .53), float3(.99, .94, .86), dust_fraction);
}

[shader("compute")]
[numthreads(4, 4, 4)]
void density_main(uint3 tid : SV_DispatchThreadID) {
    int index = tid.z / smoke_density_resolution;
    if (any(tid.xy >= uint2(smoke_density_resolution)) || index >= num_explosions) return;
    Explosion volume = explosions[index];
    uint3 cell = uint3(tid.xy, tid.z % smoke_density_resolution + volume.slot * smoke_density_resolution);
    float2 mass = transported_smoke.Load(int4(cell, 0));
    if (max(mass.x, mass.y) < .00001) {
        density_output[cell] = 0;
        return;
    }
    float3 position = voxel_position(tid, index, smoke_density_resolution);
    float age = simulation_time - volume.start_time;
    float3 drift = position - volume.foundation - age * smoke_wind;
    float3 seed = volume.seed * float3(.173, .319, .137);
    float3 q = (drift - float3(0, age * .9, 0)) * .15 + seed;
    float3 warp = float3(perlin_noise(q), perlin_noise(q + 19.1), perlin_noise(q - 13.7));
    // Erode the folds at two resolved scales as the plume develops.
    float developed = smoothstep(1, 4, age);
    float3 fold = q * 2.15 + 1.4 * warp;
    float3 curl = float3(perlin_noise(fold + 7.2), perlin_noise(fold - 11.4), perlin_noise(fold + 23.9));
    float3 displacement = (2.4 + 1.1 * developed) * warp + 1.4 * developed * curl;
    float3 displaced = position + displacement;
    float2 rolled = transported_smoke.SampleLevel(smoke_sampler,
        atlas_coordinates(displaced, volume, smoke_density_resolution), 0);
    mass = lerp(mass, rolled, .72 * smoothstep(.00001, .015, mass));
    float3 detail = (drift + displacement) * smoke_detail + seed;
    float footprint = 2 * max(volume.half_extent.x, max(volume.half_extent.y, volume.half_extent.z))
                    * smoke_detail / smoke_density_resolution;
    float soot = smoke_fbm(detail - float3(0, age * smoke_lift * smoke_detail, 0), footprint);
    float dust = smoke_fbm(detail * .78 - float3(.17, .06, .11) * age + 7.3, footprint * .78);
    float3 edge = 1 - abs((position - volume.center) / volume.half_extent);
    float boundary = smoothstep(0, .14, min(edge.x, min(edge.y, edge.z)));
    float2 contrast = float2(3.20, 3.50) + developed * float2(.65, .45);
    density_output[cell] = mass * max(.06, 1 + contrast * float2(soot, dust)) * boundary;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void lighting_main(uint3 tid : SV_DispatchThreadID) {
    int index = tid.z / smoke_light_resolution;
    if (any(tid.xy >= uint2(smoke_light_resolution)) || index >= num_explosions) return;
    uint3 cell = uint3(tid.xy, tid.z % smoke_light_resolution + explosions[index].slot * smoke_light_resolution);
    float3 position = voxel_position(tid, index, smoke_light_resolution);
    float2 components = smoke_components(position, index);
    float density = components.x + components.y;
    if (density <= .00001) {
        lighting_output[cell] = float4(0);
        return;
    }
    float dust = components.y / density;
    float3 albedo = scattering_albedo(dust);
    float g = lerp(.20, .32, dust);
    float3 eye = normalize(xform_pt(camera.cam_to_world, float3(0)) - position);

    // Two broad sky directions approximate the environment lighting integral.
    float3 sky_direction = normalize(float3(-.72, .82, -.38));
    float3 fill_direction = normalize(float3(.55, .65, .70));
    Ray sky_ray = Ray(position, sky_direction);
    Ray fill_ray = Ray(position, fill_direction);
    float sky = light_blocked(sky_ray, 220) ? 0 : smoke_transmittance(sky_ray, 150, 16);
    float fill = light_blocked(fill_ray, 220) ? 0 : smoke_transmittance(fill_ray, 150, 16);
    float3 incident = float3(.008, .013, .018)
                    + sky * phase_ratio(dot(-sky_direction, eye), g) * float3(.150, .180, .215)
                    + fill * phase_ratio(dot(-fill_direction, eye), g) * float3(.020, .035, .050);

    // Four finite patches keep nearby smoke lit across the width of each row.
    for (int i = 0; i < num_floor_fires; i++) {
        FloorFire fire = floor_fires[i];
        float area = 4 * length(cross(fire.half_u, fire.half_v)) * .56;
        float peak_radiance = max(fire.radiance.x, max(fire.radiance.y, fire.radiance.z));
        float3 emitted = fire.glow * fire.radiance / max(peak_radiance, .00001);
        for (int sample_index = 0; sample_index < 4; sample_index++) {
            float3 light_position = fire.center + ((sample_index + .5) * .5 - 1) * fire.half_u;
            float3 to_light = light_position - position;
            float distance_squared = dot(to_light, to_light);
            float distance = sqrt(max(distance_squared, .00001));
            float3 direction = to_light / distance;
            float cosine = max(0, dot(fire.normal, -direction));
            if (cosine <= 0 || fire.glow < .001) continue;
            Ray light_ray = Ray(position, direction);
            if (light_blocked(light_ray, max(.001, distance - .025))) continue;
            float transmission = smoke_transmittance(light_ray, distance, 12);
            float patch_area = area * .25;
            float solid_angle = patch_area * cosine / (distance_squared + patch_area / 3.14159265);
            incident += emitted * (solid_angle * transmission * phase_ratio(dot(-direction, eye), g)
                                  / 12.5663706);
        }
    }
    lighting_output[cell] = float4(incident * albedo * density, density);
}

// Integrate cached first-scattered radiance over a short path to the next scatter.
float3 nearby_scattered_light(Ray ray) {
    const float reach = 8;
    Hit hit;
    bool blocked = closest_scene_hit(ray, reach, shapes, num_shapes, spheres, meshes,
                                     vertices, vertex_normals, triangles, bvh_nodes, true, hit);
    float step_size = (blocked ? hit.t : reach) * .25;
    float transmission = 1;
    float3 incoming = 0;
    for (int j = 0; j < 4; j++) {
        float3 position = ray.origin + (j + .5) * step_size * ray.direction;
        float4 field = 0;
        for (int i = 0; i < num_explosions; i++) {
            Explosion volume = explosions[i];
            if (any(abs(position - volume.center) >= volume.half_extent)) continue;
            field += smoke_direct_grid.SampleLevel(smoke_sampler,
                atlas_coordinates(position, volume, smoke_light_resolution), 0);
        }
        float alpha = medium_opacity(field.w, step_size);
        incoming += transmission * alpha * field.xyz / max(field.w, .00001);
        transmission *= 1 - alpha;
    }
    return incoming;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void scatter_main(uint3 tid : SV_DispatchThreadID) {
    int res = smoke_light_resolution;
    int index = tid.z / res;
    if (any(tid.xy >= uint2(res)) || index >= num_explosions) return;
    uint3 cell = uint3(tid.xy, tid.z % res + explosions[index].slot * res);
    float4 direct = smoke_direct_grid.Load(int4(cell, 0));
    if (direct.w > .00001) {
        float3 position = voxel_position(tid, index, res);
        float dust = smoke_components(position, index).y / direct.w;
        // Tetrahedral quadrature for a diffuse second bounce. The first-bounce
        // cache supplies the same directional approximation used by reflections.
        float3 sum = 0;
        for (int k = 0; k < 4; k++) {
            float x = (k & 1) == 0 ? 1 : -1;
            float z = (k & 2) == 0 ? 1 : -1;
            float3 direction = float3(x, x * z, z) * .57735027;
            sum += nearby_scattered_light(Ray(position, direction));
        }
        direct.xyz += direct.w * scattering_albedo(dust) * sum * .25;
    }
    lighting_output[cell] = direct;
}
