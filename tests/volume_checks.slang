import geometry;
import volumes;

uniform Ray test_ray;
uniform float test_limit;
uniform float3 test_background;
RWTexture2D<float4> test_output;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main(uint3 tid : SV_DispatchThreadID) {
    float3 color = composite_explosions(test_ray, test_limit, test_background);
    float transmission = smoke_transmittance(test_ray, test_limit, 32);
    test_output[uint2(0)] = float4(color, transmission);
}
