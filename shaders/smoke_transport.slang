import volumes;

StructuredBuffer<SmokeRow> smoke_rows;
Texture3D<float2> previous_smoke;
Texture3D<float2> predicted_smoke;
RWTexture3D<float2> next_smoke;
uniform float simulation_time;
uniform float step_seconds;
uniform int history_valid;
uniform int predict_only;
uniform float3 smoke_wind;
uniform float smoke_lift;
uniform float smoke_strength;
uniform float2 smoke_half_lives;

// Undo the shared building lean. Heights below are measured from the foundation.
float3 tower_coordinates(float3 offset, float2 lean) {
    float2 c = cos(lean), s = sin(lean);
    float x = c.y * offset.x + s.y * offset.y;
    float y = -s.y * offset.x + c.y * offset.y;
    return float3(x, c.x * y + s.x * offset.z, -s.x * y + c.x * offset.z);
}

float compacted_height(float rest, float front, float ratio) {
    return ratio * min(rest, front) + max(rest - front, 0);
}

float row_height(SmokeRow row, float front, float ratio) {
    // The mesh interpolates linearly between its two transformed section ends.
    return lerp(compacted_height(row.bottom, front, ratio),
                compacted_height(row.top, front, ratio),
                (row.height - row.bottom) / (row.top - row.bottom));
}

float gaussian(float x) { return exp(-x * x); }

float2 vent_offsets(int seed, int face, int row) {
    uint h = gradient_hash(int3(seed, face, row));
    return float2(-.83, .13) + float2(h & 255, (h >> 8) & 255) * (.66 / 255);
}

float soot_source(Explosion v, float y, float wall, float along, int face, float front, float age) {
    if (wall > 9) return 0;
    float sum = 0;
    for (int i = 0; i < v.row_count; i++) {
        SmokeRow row = smoke_rows[v.first_row + i];
        float t = age - row.delay;
        if (t <= 0 || t >= 4.8) continue;
        float2 banks = vent_offsets(int(v.seed), face, i);
        float spread = .025 * max(0, wall);
        float a = gaussian((along - banks.x) / (.24 + spread));
        float b = gaussian((along - banks.y) / (.29 + spread));
        float pulse = (.66 + .45 * sin(5.1 * t + 9 * banks.x)) * a
                    + (.70 + .46 * sin(3.5 * t + 13 * banks.y)) * b;
        float envelope = smoothstep(0, .15, t) * exp(-t * .52) * (1 - smoothstep(3.5, 4.8, t));
        float height = row_height(row, front, v.compaction);
        float jet_lift = .32 + .15 * max(wall, 0) * max(wall, 0);
        float jet_width = 1.10 + .27 * max(wall, 0);
        float reach = 1.2 + 4.5 * (1 - exp(-t * 2.5));
        sum += envelope * pulse * gaussian((y - height - jet_lift) / jet_width)
             * gaussian((wall - .8) / (.55 * reach)) * (1 - smoothstep(reach, reach + 1.2, wall));
        // A short, denser puff leads the continuous flow from one window bank.
        float onset = 1 - exp(-3 * t);
        float burst = smoothstep(0, .18, t) * (1 - smoothstep(.44, 1.05, t));
        float bank = ((i + face) & 1) == 0 ? banks.x : banks.y;
        float head = gaussian((along - bank) / (.13 + .14 * onset))
                   * gaussian((wall - .6 - 4.1 * onset) / (.95 + 1.1 * onset))
                   * gaussian((y - height - .45 - 4.5 * onset) / (1.05 + 1.4 * onset));
        sum += 2.6 * burst * head;
    }
    return sum;
}

float dust_source(Explosion v, float y, float wall, float along, int face,
                  float front, float collapse_age) {
    if (collapse_age <= 0 || wall > 7) return 0;
    SmokeRow band = smoke_rows[v.first_row + v.row_count - 1];
    float height = row_height(band, front, v.compaction);
    float progress = saturate(collapse_age / v.collapse_duration);
    float settled = max(0, collapse_age - v.collapse_duration);
    float drive = smoothstep(0, .30, progress) * exp(-1.05 * settled);
    drive *= saturate(2 * (1 - v.compaction));
    float2 banks = vent_offsets(int(v.seed) + 47, face, 0);
    float bursts = .25 + .85 * gaussian((along - banks.x - .11 * sin(collapse_age * 2)) / .40)
                       + .72 * gaussian((along - banks.y) / .33);
    float roll = .77 + .23 * sin(11 * along + 3.1 * collapse_age + banks.y * 8);
    return drive * bursts * roll * gaussian((wall - 1.9) / 2.35)
         * gaussian((y - height - 1.25 - .23 * max(wall, 0)) / (1.9 + 1.4 * progress));
}

// Each velocity component is independent of its own coordinate. The eddies
// therefore add divergence-free circulation to the prescribed jet and wind.
float3 eddy_velocity(float3 p) {
    return float3(sin(.34 * p.y + .24 * p.z) + .44 * cos(.70 * p.z - .28 * p.y),
                  sin(.31 * p.z - .36 * p.x) + .46 * cos(.65 * p.x + .32 * p.z),
                  cos(.29 * p.x + .35 * p.y) + .43 * sin(.64 * p.y - .26 * p.x));
}

// In a rising billow the core overtakes the rim. Circulation in the radial
// plane turns the top outward and folds the flank back into its wake.
float3 thermal_roll(Explosion v, float3 p, int face, float age, float front, out float fresh_soot) {
    fresh_soot = 0;
    bool side = face < 2;
    float2 face_normal = side ? float2(face == 0 ? 1 : -1, 0) : float2(0, face == 2 ? 1 : -1);
    float2 tangent = side ? float2(0, 1) : float2(1, 0);
    float width = side ? v.half_size.y : v.half_size.x;
    float depth = side ? v.half_size.x : v.half_size.y;
    float stagger = vent_offsets(int(v.seed) + 19, face, 0).y;
    float spacing = 1.65 + .55 * stagger;
    float3 velocity = 0;
    for (int n = 0; n < 8; n++) {
        int i = (v.row_count - 1 + n) % v.row_count;
        SmokeRow row = smoke_rows[v.first_row + i];
        float release = row.delay + (n < 3 ? .18 + n * .88 : 2.5 + .45 * stagger + (n - 3) * spacing);
        float t = age - release;
        if (t <= 0 || t >= 6.5) continue;
        float2 banks = vent_offsets(int(v.seed), face, i);
        float along = ((n & 1) == 0 ? banks.x : banks.y) * width;
        float settled_age = max(0, release - v.collapse_delay - v.collapse_duration);
        float fuel = exp(-.14 * settled_age);
        if (n >= 3) {
            int active_face = int(gradient_hash(int3(int(v.seed), n, 41)) & 3u);
            fuel *= face == active_face ? 1 : (face == (active_face ^ 2) ? .68 : .16);
        }
        float pulse_length = 1.45 + .45 * smoothstep(0, 2.5, settled_age);
        if (n >= 3 && t < pulse_length) {
            // Feed a separate bank on each release. The gaps between banks admit backlight.
            float kick = 1 - exp(-2.8 * t);
            float outlet = row_height(row, front, v.compaction);
            float cross_jet = (dot(p.xz, tangent) - along) / (.19 * width + .65 * kick);
            float away = (dot(p.xz, face_normal) - depth - .65 - 3.5 * kick) / (1.05 + 1.2 * kick);
            float up = (p.y - outlet - .35 - 3.8 * kick) / (1.0 + 1.35 * kick);
            float pulse = smoothstep(0, .20, t) * (1 - smoothstep(.67, pulse_length, t));
            fresh_soot += 1.55 * fuel * pulse * exp(-cross_jet * cross_jet - away * away - up * up);
        }
        float progress = saturate((release - v.collapse_delay) / v.collapse_duration);
        float y = row_height(row, v.building_height * progress * progress, v.compaction);
        float2 center = tangent * along + face_normal * (depth + 1 + 5.1 * (1 - exp(-t)))
                      + t * smoke_wind.xz;
        float2 radial = p.xz - center;
        float r = length(radial);
        float radius = 1.45 + 5.4 * (1 - exp(-.22 * t));
        float core = .72 + .32 * radius;
        float rise = smoke_lift * (.92 * t + 2.8 * (1 - exp(-t / 2.8)));
        float dy = (p.y - y - 1 - rise) / core;
        float dr = (r - radius) / core;
        float strength = 7.5 * exp(-.68 * (dr * dr + dy * dy))
                       * smoothstep(0, .4, t) * (1 - smoothstep(4, 6.5, t)) * fuel;
        float2 horizontal = radial * (dy / max(r, .1));
        velocity += strength * float3(horizontal.x, -dr, horizontal.y);
    }
    return velocity;
}

float2 transported_sample(Texture3D<float2> field, float3 position, Explosion v) {
    if (any(abs(position - v.center) >= v.half_extent)) return 0;
    return field.SampleLevel(smoke_sampler,
        atlas_coordinates(position, v, smoke_density_resolution), 0);
}

// Bound the MacCormack correction by the donors of the trilinear backtrace.
// This keeps a thin wisp sharp without creating negative density at its edge.
float limited_density(float value, float3 departure, Explosion v, int channel) {
    if (any(abs(departure - v.center) >= v.half_extent)) return 0;
    int res = smoke_density_resolution;
    float3 uvw = .5 + (departure - v.center) / (2 * v.half_extent);
    int3 base = int3(floor(uvw * res - .5));
    float low = 1e30, high = 0;
    [unroll]
    for (int k = 0; k < 8; k++) {
        int3 cell = clamp(base + int3(k & 1, (k >> 1) & 1, k >> 2), 0, res - 1);
        cell.z += v.slot * res;
        float donor = previous_smoke.Load(int4(cell, 0))[channel];
        low = min(low, donor);
        high = max(high, donor);
    }
    return clamp(value, low, high);
}

[shader("compute")]
[numthreads(4, 4, 4)]
void advect_main(uint3 tid : SV_DispatchThreadID) {
    int res = smoke_density_resolution;
    int index = tid.z / res;
    if (any(tid.xy >= uint2(res)) || index >= num_explosions) return;
    Explosion v = explosions[index];
    uint3 cell = uint3(tid.xy, tid.z % res);
    uint3 target = uint3(cell.xy, cell.z + res * v.slot);
    float end_age = simulation_time - v.start_time;
    if (end_age <= 0 || smoke_strength == 0) {
        next_smoke[target] = 0;
        return;
    }
    float dt = min(step_seconds, end_age);
    float age = end_age - .5 * dt;
    float collapse_age = age - v.collapse_delay;
    float progress = saturate(collapse_age / v.collapse_duration);
    float front = v.building_height * progress * progress;
    float3 uvw = (float3(cell) + .5) / res * 2 - 1;
    float3 world = v.center + uvw * v.half_extent;
    float3 p = tower_coordinates(world - v.foundation, v.lean * smoothstep(0, 1, progress));
    float2 side = abs(p.xz) - v.half_size;
    float wall = length(max(side, 0)) + min(max(side.x, side.y), 0);
    float roof = compacted_height(v.building_height, front, v.compaction);
    float outside = smoothstep(-.10, .50, max(wall, p.y - roof));
    if (outside <= 0 || p.y <= 0) {
        next_smoke[target] = 0;
        return;
    }
    float side_weight = smoothstep(-1.8, 1.8, side.x - side.y);
    float2 outward = sign(p.xz) * float2(side_weight, 1 - side_weight);
    outward /= max(length(outward), .001);
    float3 normal = float3(outward.x, 0, outward.y);
    bool side_face = side.x > side.y;
    int face = side_face ? (p.x > 0 ? 0 : 1) : (p.z > 0 ? 2 : 3);
    float along = side_face ? p.z / v.half_size.y : p.x / v.half_size.x;

    float3 drift = world - v.foundation - (smoke_wind + float3(0, .65, 0)) * age;
    float3 eddy = eddy_velocity(drift + v.seed * float3(.61, .19, .43));
    float wall_jet = exp(-max(wall, 0) * .18);
    SmokeRow upper_row = smoke_rows[v.first_row + v.row_count - 1];
    float outlet = row_height(upper_row, front, v.compaction);
    float cooled = smoothstep(6, 25, max(0, p.y - outlet) + .45 * max(0, wall));
    float3 broad_eddy = eddy_velocity(drift * .56 + v.seed * float3(.61, .19, .43));
    float3 soot_eddy = lerp(eddy, broad_eddy, .42 * cooled);
    float fresh_soot;
    float3 roll = thermal_roll(v, p, face, age, front, fresh_soot);
    float3 soot_velocity = smoke_wind * (1 + .55 * cooled) + soot_eddy * 2.15 + roll
                         + normal * (.40 + 6.6 * wall_jet * (1 - .78 * cooled))
                         + float3(0, smoke_lift * lerp(2.18 - .55 * wall_jet, .86, cooled), 0);
    float spreading = (1 - .53 * smoothstep(9, 25, wall))
                    * (1 - .65 * smoothstep(.4, 4.5, collapse_age - v.collapse_duration));
    float3 dust_velocity = .85 * smoke_wind + eddy * 2.05
                         + normal * (1.45 + 7.1 * exp(-max(wall, 0) * .075)) * spreading;
    dust_velocity.y += 1.42 + 2.0 * wall_jet - 1.68 * smoothstep(4, 8, collapse_age);

    float3 soot_departure = world - dt * soot_velocity;
    float3 dust_departure = world - dt * dust_velocity;
    bool has_mass = history_valid != 0 && end_age > step_seconds;
    if (predict_only != 0) {
        float2 prediction = 0;
        if (has_mass) {
            prediction.x = transported_sample(previous_smoke, soot_departure, v).x;
            prediction.y = transported_sample(previous_smoke, dust_departure, v).y;
        }
        next_smoke[target] = prediction;
        return;
    }
    float2 density = 0;
    if (has_mass) {
        float2 prediction = predicted_smoke.Load(int4(target, 0));
        float2 original = previous_smoke.Load(int4(target, 0));
        if (max(prediction.x, original.x) > .000001) {
            float reverse = transported_sample(predicted_smoke, world + dt * soot_velocity, v).x;
            density.x = limited_density(prediction.x + .5 * (original.x - reverse),
                                        soot_departure, v, 0);
        }
        if (max(prediction.y, original.y) > .000001) {
            float reverse = transported_sample(predicted_smoke, world + dt * dust_velocity, v).y;
            density.y = limited_density(prediction.y + .5 * (original.y - reverse),
                                        dust_departure, v, 1);
        }
        // Dilute edges mix with clear air faster than the cores of the billows.
        float2 core = smoothstep(.07, .43, density / smoke_strength);
        density *= exp2(-dt / (smoke_half_lives * lerp(.16, 1.0, core)));
    }
    density.x += dt * smoke_strength * 7.8 * (soot_source(v, p.y, wall, along, face, front, age) + fresh_soot);
    density.y += dt * smoke_strength * 4.3 * dust_source(v, p.y, wall, along, face, front, collapse_age);
    next_smoke[target] = density * outside;
}
