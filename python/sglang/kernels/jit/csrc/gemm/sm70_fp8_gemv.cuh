#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp16.h>
#include <limits>

namespace sglang {

static_assert(SGL_CUDA_ARCH == 700, "Packed FP8 vector GEMV requires SM70");

namespace fp8_gemv {

template <bool Fused>
__device__ __forceinline__ float accumulate(float value, float weight, float sum) {
  // Wider verification blocks are arithmetic limited. Both multiplicands are
  // FP16 values, so their product is exact in FP32; FMA saves an instruction.
  // Keep the qualified <=4-row reduction order unchanged.
  if constexpr (Fused)
    return __fmaf_rn(value, weight, sum);
  else
    return __fadd_rn(sum, __fmul_rn(value, weight));
}

template <bool Aligned>
__device__ __forceinline__ half2 load_scale_pair(const half* p) {
  if constexpr (Aligned)
    return *reinterpret_cast<const half2*>(p);
  else
    return __halves2half2(p[0], p[1]);
}

/// Decode four E4M3 bytes together, retaining FP16 operands and FP32 partials.
template <int Rows, int Group, bool Aligned>
__global__ void vector_partial(
    const half* __restrict__ x,
    const uint32_t* __restrict__ w,
    const half* __restrict__ scales,
    float* __restrict__ p,
    uint32_t n,
    uint32_t k,
    uint32_t pn,
    uint32_t pk,
    uint32_t macro) {
  constexpr int BN = 64, KL = 8, NW = BN / 4;
  static_assert(Rows >= 1 && Rows <= 8 && (Group == 32 || Group == 128));
  const uint32_t word_lane = threadIdx.x % NW;
  const uint32_t k_lane = threadIdx.x / NW;
  const uint32_t tiles = macro / 64;
  const uint32_t ntile = blockIdx.x / (macro / BN);
  const uint32_t sub = blockIdx.x % (macro / BN);
  const uint32_t col = ntile * macro + (word_lane % tiles) * 64 + sub * (BN / tiles) + 4 * (word_lane / tiles);
  const uint32_t first_k = blockIdx.y * 128;
  float accum[Rows][4] = {};
#pragma unroll
  for (int group = 0; group < 128; group += Group) {
    const uint32_t scale_at = ((first_k + group) / Group) * pn + col;
    const half2 s01 = load_scale_pair<Aligned>(scales + scale_at);
    const half2 s23 = load_scale_pair<Aligned>(scales + scale_at + 2);
#pragma unroll
    for (int step = 0; step < Group / KL; ++step) {
      const uint32_t kk = first_k + group + step * KL + k_lane;
      const uint32_t at =
          (kk / 16) * pn * 4 + ntile * tiles * 256 + ((kk % 16) * 16 + (col % 64) / 4) * tiles + (col / 64) % tiles;
      const uint32_t packed = w[at], high = packed >> 8;
      const uint32_t b01 = ((packed & 0x007f007fu) << 7) | ((packed & 0x00800080u) << 8);
      const uint32_t b23 = ((high & 0x007f007fu) << 7) | ((high & 0x00800080u) << 8);
      const half2 h01 = __halves2half2(__ushort_as_half(b01 & 65535), __ushort_as_half(b01 >> 16));
      const half2 h23 = __halves2half2(__ushort_as_half(b23 & 65535), __ushort_as_half(b23 >> 16));
      // Half multiplication gives the same rounded operand as the original
      // FP32 scale product followed by FP16 conversion. Products with X and
      // all K accumulation below stay FP32, including values beyond FP16 range.
      const float2 f01 = __half22float2(__hmul2(h01, s01));
      const float2 f23 = __half22float2(__hmul2(h23, s23));
#pragma unroll
      for (int row = 0; row < Rows; ++row) {
        const float value = kk < k ? __half2float(x[row * k + kk]) : 0.0f;
        accum[row][0] = accumulate<(Rows >= 5 && Group == 32)>(value, f01.x, accum[row][0]);
        accum[row][1] = accumulate<(Rows >= 5 && Group == 32)>(value, f01.y, accum[row][1]);
        accum[row][2] = accumulate<(Rows >= 5 && Group == 32)>(value, f23.x, accum[row][2]);
        accum[row][3] = accumulate<(Rows >= 5 && Group == 32)>(value, f23.y, accum[row][3]);
      }
    }
  }
  __shared__ __align__(16) float sums[Rows][KL][BN];
#pragma unroll
  for (int row = 0; row < Rows; ++row)
    *reinterpret_cast<float4*>(&sums[row][k_lane][word_lane * 4]) =
        make_float4(accum[row][0], accum[row][1], accum[row][2], accum[row][3]);
  __syncthreads();
  if (k_lane == 0) {
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
      float4 value = *reinterpret_cast<float4*>(&sums[row][0][word_lane * 4]);
#pragma unroll
      for (int split = 1; split < KL; ++split) {
        const float4 v = *reinterpret_cast<float4*>(&sums[row][split][word_lane * 4]);
        value.x = __fadd_rn(value.x, v.x);
        value.y = __fadd_rn(value.y, v.y);
        value.z = __fadd_rn(value.z, v.z);
        value.w = __fadd_rn(value.w, v.w);
      }
      const float values[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
      for (int c = 0; c < 4; ++c)
        if (col + c < n) p[(row * (pk / 128) + blockIdx.y) * n + col + c] = values[c];
    }
  }
}
}  // namespace fp8_gemv

/// Validate a packed Marlin projection and launch its FP32 partial contraction.
template <int Rows, int Group>
void sm70_fp8_vector_run(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView scales,
    tvm::ffi::TensorView partials,
    tvm::ffi::TensorView output) {
  using namespace host;
  SymbolicSize k{"k"}, n{"n"}, pn{"pn"}, pk16{"pk16"}, groups{"groups"}, packed_n{"packed_n"};
  SymbolicDevice device;
  TensorMatcher({Rows, k}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(x);
  TensorMatcher({groups, pn}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(scales);
  TensorMatcher({pk16, packed_n}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(w);
  TensorMatcher({Rows, n}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(output);
  const auto pk = pk16.unwrap() * 16;
  TensorMatcher({Rows, pk / 128, n}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(partials);
  CHECK_HOST(pk > 0 && pk % 128 == 0 && groups.unwrap() * Group == pk && pn.unwrap() % 128 == 0);
  CHECK_HOST(packed_n.unwrap() == pn.unwrap() * 4);
  CHECK_HOST(0 < k.unwrap() && k.unwrap() <= pk && 0 < n.unwrap() && n.unwrap() <= pn.unwrap());
  CHECK_HOST(static_cast<uint64_t>(pk) * pn.unwrap() <= std::numeric_limits<uint32_t>::max());
  CHECK_HOST(static_cast<uint64_t>(Rows) * pk / 128 * n.unwrap() <= std::numeric_limits<uint32_t>::max());
  const auto kernel = reinterpret_cast<uintptr_t>(scales.data_ptr()) % 4 == 0
                          ? fp8_gemv::vector_partial<Rows, Group, true>
                          : fp8_gemv::vector_partial<Rows, Group, false>;
  LaunchKernel(dim3(pn.unwrap() / 64, pk / 128), 128, device.unwrap())(
      kernel,
      static_cast<const half*>(x.data_ptr()),
      static_cast<const uint32_t*>(w.data_ptr()),
      static_cast<const half*>(scales.data_ptr()),
      static_cast<float*>(partials.data_ptr()),
      static_cast<uint32_t>(n.unwrap()),
      static_cast<uint32_t>(k.unwrap()),
      static_cast<uint32_t>(pn.unwrap()),
      static_cast<uint32_t>(pk),
      pn.unwrap() % 256 == 0 ? 256u : 128u);
}
}  // namespace sglang
