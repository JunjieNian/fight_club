import geometry;

struct Explosion {
    float3 center;
    float3 half_extent;
    float3 foundation;
    float2 half_size;
    float building_height;
    int slot;
    int first_row;
    int row_count;
    float start_time;
    float seed;
    float collapse_delay;
    float collapse_duration;
    float compaction;
    float2 lean;
};

struct SmokeRow {
    float height;
    float bottom;
    float top;
    float delay;
};

uniform int num_explosions;
uniform int smoke_capacity;
uniform int smoke_density_resolution;
uniform int smoke_light_resolution;
uniform float smoke_step_size;
uniform float smoke_albedo;
StructuredBuffer<Explosion> explosions;
Texture3D<float2> smoke_density_grid;
Texture3D<float4> smoke_lighting_grid;
SamplerState smoke_sampler;

uint gradient_hash(int3 cell) {
    uint3 p = uint3(cell);
    uint h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u ^ p.z * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return h;
}

float gradient_dot(int3 cell, float3 offset) {
    uint h = gradient_hash(cell) & 15;
    float u = h < 8 ? offset.x : offset.y;
    float v = h < 4 ? offset.y : (h == 12 || h == 14 ? offset.x : offset.z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

float perlin_noise(float3 p) {
    int3 cell = int3(floor(p));
    float3 f = frac(p);
    float3 w = f * f * f * (f * (f * 6 - 15) + 10);
    float lower = lerp(lerp(gradient_dot(cell, f),
                            gradient_dot(cell + int3(1, 0, 0), f - float3(1, 0, 0)), w.x),
                       lerp(gradient_dot(cell + int3(0, 1, 0), f - float3(0, 1, 0)),
                            gradient_dot(cell + int3(1, 1, 0), f - float3(1, 1, 0)), w.x), w.y);
    float upper = lerp(lerp(gradient_dot(cell + int3(0, 0, 1), f - float3(0, 0, 1)),
                            gradient_dot(cell + int3(1, 0, 1), f - float3(1, 0, 1)), w.x),
                       lerp(gradient_dot(cell + int3(0, 1, 1), f - float3(0, 1, 1)),
                            gradient_dot(cell + int3(1, 1, 1), f - float3(1, 1, 1)), w.x), w.y);
    return lerp(lower, upper, w.z);
}

float smoke_fbm(float3 p, float footprint) {
    float value = 0;
    float amplitude = .56;
    for (int octave = 0; octave < 5; octave++) {
        float filtered_amplitude = amplitude * (1 - smoothstep(.35, .8, footprint));
        value += filtered_amplitude * perlin_noise(p);
        p = p * 2.03 + float3(7.1, 19.3, -5.7);
        amplitude *= .49;
        footprint *= 2.03;
    }
    return value;
}

bool volume_interval(Ray ray, Explosion volume, float limit, out float begin, out float end) {
    begin = 0;
    end = limit;
    for (int axis = 0; axis < 3; axis++) {
        float lower = volume.center[axis] - volume.half_extent[axis];
        float upper = volume.center[axis] + volume.half_extent[axis];
        if (abs(ray.direction[axis]) < 1e-7) {
            if (ray.origin[axis] < lower || ray.origin[axis] > upper) return false;
        } else {
            float a = (lower - ray.origin[axis]) / ray.direction[axis];
            float b = (upper - ray.origin[axis]) / ray.direction[axis];
            begin = max(begin, min(a, b));
            end = min(end, max(a, b));
        }
    }
    return end > begin;
}

float3 atlas_coordinates(float3 position, Explosion volume, int resolution) {
    float3 uv = (position - volume.center) / (2 * volume.half_extent) + .5;
    uv = clamp(uv, .5 / resolution, 1 - .5 / resolution);
    uv.z = (uv.z + volume.slot) / smoke_capacity;
    return uv;
}

float2 smoke_components(float3 position, int index) {
    Explosion volume = explosions[index];
    if (any(abs(position - volume.center) >= volume.half_extent)) return 0;
    return smoke_density_grid.SampleLevel(smoke_sampler,
        atlas_coordinates(position, volume, smoke_density_resolution), 0);
}

float sample_smoke_density(float3 position, int index) {
    float2 material = smoke_components(position, index);
    return material.x + material.y;
}

// Henyey-Greenstein relative to the isotropic phase value of 1 / (4 pi).
float phase_ratio(float cosine, float g) {
    float denominator = max(1e-5, 1 + g * g - 2 * g * clamp(cosine, -1, 1));
    return (1 - g * g) / (denominator * sqrt(denominator));
}

float smoke_transmittance(Ray ray, float limit, int steps) {
    float optical_depth = 0;
    for (int i = 0; i < num_explosions; i++) {
        float begin, end;
        if (!volume_interval(ray, explosions[i], limit, begin, end)) continue;
        float step_size = (end - begin) / steps;
        for (int j = 0; j < steps; j++) {
            float3 position = ray.origin + (begin + (j + .5) * step_size) * ray.direction;
            optical_depth += sample_smoke_density(position, i) * step_size;
        }
    }
    return exp(-optical_depth);
}

float medium_opacity(float extinction, float distance) {
    return 1 - exp(-max(0, extinction) * max(0, distance));
}

float3 composite_explosions(Ray ray, float limit, float3 background) {
    float begin = limit;
    float end = 0;
    for (int i = 0; i < num_explosions; i++) {
        float near_t, far_t;
        if (volume_interval(ray, explosions[i], limit, near_t, far_t)) {
            begin = min(begin, near_t);
            end = max(end, far_t);
        }
    }
    if (end <= begin) return background;
    int steps = clamp(int(ceil((end - begin) / smoke_step_size)), 1, 384);
    float step_size = (end - begin) / steps;
    float transmission = 1;
    float3 accumulated = float3(0);
    for (int sample_index = 0; sample_index < steps; sample_index++) {
        float3 position = ray.origin + (begin + (sample_index + .5) * step_size) * ray.direction;
        float extinction = 0;
        float3 source = float3(0);
        for (int i = 0; i < num_explosions; i++) {
            float density = sample_smoke_density(position, i);
            if (density <= .00001) continue;
            // Premultiplied interpolation preserves illumination at thin edges.
            float4 lit = smoke_lighting_grid.SampleLevel(smoke_sampler,
                atlas_coordinates(position, explosions[i], smoke_light_resolution), 0);
            float3 color = lit.xyz / max(lit.w, .00001);
            float haze = 1 - exp(-max(0, -position.z - 20) * .0022);
            color = lerp(color, float3(.014, .035, .047), haze);
            extinction += density;
            source += density * color;
        }
        float alpha = medium_opacity(extinction, step_size);
        accumulated += transmission * alpha * source / max(extinction, 1e-6);
        transmission *= 1 - alpha;
        if (transmission < .001) break;
    }
    return accumulated + transmission * background;
}
