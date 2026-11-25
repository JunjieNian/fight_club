import geometry;
import volumes;
import fight_club;

RWTexture2D<float2> measured;

[shader("compute")]
[numthreads(8, 8, 1)]
void measure_main(uint2 tid : SV_DispatchThreadID) {
    if (any(tid >= uint2(resolution))) return;
    Ray ray = camera_ray(tid, uint(resolution.x), uint(resolution.y), camera);
    Hit hit;
    float limit = scene_hit(ray, RAY_MAX, hit) ? hit.t : RAY_MAX;
    Explosion tower = explosions[0];
    float depth = (tower.foundation.z - ray.origin.z) / ray.direction.z;
    float3 local = ray.origin + depth * ray.direction - tower.foundation;
    // Sample the former tower and its flanks through the open window.
    bool in_view = limit > 30 && abs(local.x) < 1.9 * tower.half_size.x
                && local.y > .25 * tower.building_height && local.y < tower.building_height;
    measured[tid] = float2(smoke_transmittance(ray, limit, 128), in_view ? 1 : 0);
}
