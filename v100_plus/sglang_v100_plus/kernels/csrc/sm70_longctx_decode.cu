// SPDX-License-Identifier: Apache-2.0
// Long-context grouped decode for Volta (SM70), D256 G3/G6 (H3|6/Hkv1), FP16 or
// E5M2 KV.
//
// Read-once split-KV design: grid = (kv_heads, splits, batch). Each CTA owns
// one kv head and one context partition, streaming that partition's K/V once
// with 16-byte vectorized loads from the dense token-major layout. This keeps
// DRAM traffic at the theoretical minimum (unlike per-query-head partition
// schemes that re-read the same KV once per GQA head). A split-merge kernel
// (the existing TileLang combine) rescales the FP32 softmax states, with the
// same partial-output ABI. Numerics mirror _kernels_paged_decode.py: scores
// are scaled by softmax_scale*k_scale, probabilities are
// exp2((s-m)*scale*log2e), and the partial output is normalized by the
// partition sum and v_scale.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace sm70_longctx {

constexpr int kThreads = 512;
constexpr int kBlockN = 32; // tokens per tile (smem-bounded)
constexpr int kDim = 256;
constexpr int kPairsPerLane = (kDim / 2) / 32; // half2 pairs per lane (4)
constexpr int kAccPerLane = 2 * kPairsPerLane; // fp32 accumulators per lane
constexpr int kKVStride = 264; // padded smem row stride (half units)
constexpr int kSmemKV = kBlockN * kKVStride; // halves per K or V tile
constexpr float kLog2E = 1.4426950408889634f;

constexpr int kIndexerThreads = 256;
constexpr int kIndexerWarps = kIndexerThreads / 32;
constexpr int kIndexerHeads = 4;
constexpr int kIndexerDim = 128;
constexpr int kIndexerKeysPerWarp = 8;
constexpr int kIndexerKeysPerBlock = kIndexerWarps * kIndexerKeysPerWarp;

__device__ __forceinline__ void e5m2_to_fp16_16(const uint4 raw, half *dst) {
  // E5M2 shares sign/exponent bit positions with FP16: each byte shifted left
  // by 8 is an exact value conversion, and the 16 output halves must keep the
  // same dim order as the 16 input bytes. Each 32-bit input word b = [b0..b3]
  // produces one uint32 pair [fp16(b0), fp16(b1)] and one [fp16(b2), fp16(b3)];
  // the pairs go to consecutive uint32 slots so dim order is preserved.
  const uint32_t *r = reinterpret_cast<const uint32_t *>(&raw);
  uint32_t out[8];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const uint32_t b = r[w];
    out[2 * w] = (b & 0x000000ffu) << 8 | ((b >> 8) & 0x000000ffu) << 24;
    out[2 * w + 1] = ((b >> 16) & 0x000000ffu) << 8 | ((b >> 24) & 0x000000ffu)
                                                          << 24;
  }
  *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(out);
  *reinterpret_cast<uint4 *>(dst + 8) =
      *reinterpret_cast<const uint4 *>(out + 4);
}

// Copy one aligned cache vector into the FP16 shared tile. FP16 storage is
// loaded without conversion; the legacy byte specialization decodes E5M2.
template <typename CacheType>
__device__ __forceinline__ void load_cache_vector(const CacheType *cache,
                                                  int64_t offset, half *dst,
                                                  bool valid) {
  static_assert(std::is_same_v<CacheType, __half> ||
                std::is_same_v<CacheType, uint8_t>);
  uint4 raw = make_uint4(0, 0, 0, 0);
  if (valid) {
    raw = *reinterpret_cast<const uint4 *>(cache + offset);
  }
  if constexpr (std::is_same_v<CacheType, __half>) {
    *reinterpret_cast<uint4 *>(dst) = raw;
  } else {
    e5m2_to_fp16_16(raw, dst);
  }
}

// Split-KV QSA decode for the TP4/TP8 H6|3/Hkv1/D256 layouts.  Unlike the
// generic fallback, this resolves the selected logical positions while loading
// the cache and keeps K/V shared by all local query heads.  No FP16 compact-KV
// scratch is written or read.
template <int Group, typename CacheType>
__global__ void __launch_bounds__(kThreads, 1) qsa_decode_partial_kernel(
    const __half *__restrict__ q, const CacheType *__restrict__ k_cache,
    const CacheType *__restrict__ v_cache, const int *__restrict__ req_to_token,
    const int *__restrict__ req_indices, const int *__restrict__ indices,
    const int *__restrict__ seq_lens, const int req_stride, const int topk,
    const int max_splits, const int min_tokens_per_split,
    const float score_scale, __half *__restrict__ partial_o,
    float *__restrict__ partial_lse) {
  constexpr int kGroup = Group;
  const int split_id = blockIdx.y;
  const int seq_id = blockIdx.z;
  const int seq_len = seq_lens[seq_id];
  const int context = min(seq_len, topk);
  const int active_splits =
      min(max_splits,
          max(1, (context + min_tokens_per_split - 1) / min_tokens_per_split));
  if (split_id >= active_splits || context <= 0) {
    return;
  }
  // Balanced boundaries keep every active split nonempty.
  const int split_begin = context * split_id / active_splits;
  const int split_end = context * (split_id + 1) / active_splits;

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];
  __shared__ int slots[kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;
  const int req_index = req_indices[seq_id];
  const float scale_log2 = score_scale * kLog2E;

  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    qs[i] = q[(int64_t)seq_id * kGroup * kDim + i];
  }

  // Each compute warp owns one query head. Thread-local arrays indexed by
  // the runtime warp ID force these accumulators into local memory on SM70.
  // Keep only this thread's head state so ptxas can retain it in registers.
  float m_row = -1.0e30f;
  float l_row = 0.f;
  float o_acc[kAccPerLane];
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (split_end - split_begin + kBlockN - 1) / kBlockN;
  constexpr int kVectorElems = sizeof(uint4) / sizeof(CacheType);
  constexpr int kLoadIters =
      (kBlockN * (kDim / kVectorElems) + kThreads - 1) / kThreads;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = split_begin + tile * kBlockN;
    const int tile_tokens = min(kBlockN, split_end - tile_begin);
    if (tid < kBlockN) {
      const int selected = tile_begin + tid;
      int slot = -1;
      if (selected < split_end) {
        const int logical = indices[(int64_t)seq_id * topk + selected];
        if (logical >= 0 && logical < seq_len) {
          slot = req_to_token[(int64_t)req_index * req_stride + logical];
        }
      }
      slots[tid] = slot;
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < kLoadIters; ++it) {
      const int vector = tid + it * kThreads;
      const int element_off = vector * kVectorElems;
      const int tok_local = element_off / kDim;
      const int d_off = element_off % kDim;
      const int slot = slots[tok_local];
      const bool valid = tok_local < tile_tokens && slot >= 0;
      const int64_t base = (int64_t)slot * kDim + d_off;
      const int dst = tok_local * kKVStride + d_off;
      load_cache_vector(k_cache, base, ks + dst, valid);
      load_cache_vector(v_cache, base, vs + dst, valid);
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
      const __half2 *qr = reinterpret_cast<const __half2 *>(qs + r * kDim);
      const int n = lane;
      const __half2 *krow =
          reinterpret_cast<const __half2 *>(ks + n * kKVStride);
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
      for (int h = 0; h < kDim / 8; ++h) {
        const __half2 qh0 = qr[4 * h];
        const __half2 kh0 = krow[4 * h];
        const __half2 qh1 = qr[4 * h + 1];
        const __half2 kh1 = krow[4 * h + 1];
        const __half2 qh2 = qr[4 * h + 2];
        const __half2 kh2 = krow[4 * h + 2];
        const __half2 qh3 = qr[4 * h + 3];
        const __half2 kh3 = krow[4 * h + 3];
        acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
        acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
        acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
        acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
        acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
        acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
        acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
        acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
      }
      scores[r * kBlockN + n] = (n < tile_tokens && slots[n] >= 0)
                                    ? acc0 + acc1 + acc2 + acc3
                                    : -1.0e30f;
      __syncwarp();

      float loc_max = scores[r * kBlockN + lane];
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row, loc_max);
      const float alpha = exp2f((m_row - m_new) * scale_log2);
      m_row = m_new;
      l_row *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[j] *= alpha;
        }
      }
      float p = 0.f;
      if (lane < tile_tokens && slots[lane] >= 0) {
        p = exp2f((scores[r * kBlockN + lane] - m_new) * scale_log2);
      }
      probs[r * kBlockN + lane] = __float2half(p);
      float loc_sum = p;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row += loc_sum;
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2 *vrow =
            reinterpret_cast<const __half2 *>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 value = vrow[lane + j * 32];
          o_acc[2 * j] = fmaf(p, __half2float(value.x), o_acc[2 * j]);
          o_acc[2 * j + 1] = fmaf(p, __half2float(value.y), o_acc[2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row > 0.f ? 1.f / l_row : 0.f;
    __half *out_row =
        partial_o +
        (((int64_t)seq_id * max_splits + split_id) * kGroup + r) * kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      out_row[2 * (lane + j * 32)] = __float2half_rn(o_acc[2 * j] * inv_l);
      out_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[2 * j + 1] * inv_l);
    }
    if (lane == 0) {
      partial_lse[((int64_t)seq_id * max_splits + split_id) * kGroup + r] =
          l_row > 0.f ? __log2f(l_row) + m_row * scale_log2 : -1.0e30f;
    }
  }
}

// QSA indexer decode scoring for Qwen3.8.  A warp owns one compressed key,
// reads its 128 FP16 values once, and accumulates the four query-head scores
// in registers.  This avoids padding four real heads to Volta MMA's 16-head
// tile and emits the exact FP32 logits consumed by fast_topk.
template <int PageSize>
__global__ void __launch_bounds__(kIndexerThreads, 2)
    qsa_indexer_decode_kernel(const __half *__restrict__ q,
                              const __half *__restrict__ k_cache,
                              const int *__restrict__ page_table,
                              const int *__restrict__ context_lens,
                              const int max_pages, const int max_model_len,
                              const float score_scale,
                              float *__restrict__ logits) {
  const int seq_id = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int context = context_lens[seq_id];
  const int block_begin = blockIdx.x * kIndexerKeysPerBlock;
  if (block_begin >= context) {
    return;
  }

  __shared__ __half qs[kIndexerHeads * kIndexerDim];
  for (int i = tid; i < kIndexerHeads * kIndexerDim; i += kIndexerThreads) {
    qs[i] = q[(int64_t)seq_id * kIndexerHeads * kIndexerDim + i];
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < kIndexerKeysPerWarp; ++it) {
    const int position = block_begin + it * kIndexerWarps + warp;
    if (position >= context || position >= max_model_len) {
      continue;
    }
    const int logical_page = position / PageSize;
    const int page_offset = position % PageSize;
    const int physical_page =
        page_table[(int64_t)seq_id * max_pages + logical_page];
    const __half2 *key = reinterpret_cast<const __half2 *>(
        k_cache +
        ((int64_t)physical_page * PageSize + page_offset) * kIndexerDim);
    const __half2 k0 = key[lane * 2];
    const __half2 k1 = key[lane * 2 + 1];
    float scores[kIndexerHeads];
#pragma unroll
    for (int h = 0; h < kIndexerHeads; ++h) {
      const __half2 *qh =
          reinterpret_cast<const __half2 *>(qs + h * kIndexerDim);
      const __half2 q0 = qh[lane * 2];
      const __half2 q1 = qh[lane * 2 + 1];
      float acc = 0.f;
      acc = fmaf(__half2float(q0.x), __half2float(k0.x), acc);
      acc = fmaf(__half2float(q0.y), __half2float(k0.y), acc);
      acc = fmaf(__half2float(q1.x), __half2float(k1.x), acc);
      acc = fmaf(__half2float(q1.y), __half2float(k1.y), acc);
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
      }
      scores[h] = acc;
    }
    if (lane == 0) {
      float score = 0.f;
#pragma unroll
      for (int h = 0; h < kIndexerHeads; ++h) {
        score += fmaxf(scores[h], 0.f);
      }
      logits[(int64_t)seq_id * max_model_len + position] = score / score_scale;
    }
  }
}

// QSA chunk-prefill for the exact TP4 full-attention shape.  One CTA owns one
// query token and evaluates all local GQA heads together, so every selected
// K/V row is fetched once instead of once per query head.  The
// logical QSA indices are resolved through req_to_token in the load phase;
// this avoids materializing and converting the entire accumulated context on
// every 8K serving chunk.
template <int Group, typename CacheType>
__global__ void __launch_bounds__(kThreads, 1) qsa_prefill_kernel(
    const __half *__restrict__ q, const CacheType *__restrict__ k_cache,
    const CacheType *__restrict__ v_cache, const int *__restrict__ req_to_token,
    const int *__restrict__ req_indices, const int *__restrict__ indices,
    const int *__restrict__ seq_lens, const int req_stride, const int topk,
    const float score_scale, __half *__restrict__ output) {
  constexpr int kGroup = Group;
  const int query = blockIdx.x;
  const int req_index = req_indices[0];
  const int seq_len = seq_lens[0];
  // seq_len includes the complete serving chunk.  QSA stores each row's
  // causal candidates first, so mirror _sparse_gqa_chunk_prefill and consume
  // only the prefix visible to this query row.
  const int visible = min(seq_len, query + seq_len - gridDim.x + 1);
  const int row_topk = min(topk, max(0, visible));

  __shared__ __half ks[kSmemKV];
  __shared__ __half vs[kSmemKV];
  __shared__ __half qs[kGroup * kDim];
  __shared__ float scores[kGroup * kBlockN];
  __shared__ __half probs[kGroup * kBlockN];
  __shared__ int slots[kBlockN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const bool is_compute_warp = warp < kGroup;
  const float scale_log2 = score_scale * kLog2E;

  for (int i = tid; i < kGroup * kDim; i += kThreads) {
    qs[i] = q[(int64_t)query * kGroup * kDim + i];
  }

  float m_row[kGroup];
  float l_row[kGroup];
  float o_acc[kGroup][kAccPerLane];
#pragma unroll
  for (int r = 0; r < kGroup; ++r) {
    m_row[r] = -1.0e30f;
    l_row[r] = 0.f;
  }
  if (is_compute_warp) {
#pragma unroll
    for (int j = 0; j < kAccPerLane; ++j) {
      o_acc[warp][j] = 0.f;
    }
  }
  __syncthreads();

  const int num_tiles = (row_topk + kBlockN - 1) / kBlockN;
  constexpr int kVectorElems = sizeof(uint4) / sizeof(CacheType);
  constexpr int kLoadIters =
      (kBlockN * (kDim / kVectorElems) + kThreads - 1) / kThreads;
  for (int tile = 0; tile < num_tiles; ++tile) {
    const int tile_begin = tile * kBlockN;
    const int tile_tokens = min(kBlockN, row_topk - tile_begin);
    if (tid < kBlockN) {
      const int selected = tile_begin + tid;
      int slot = -1;
      if (selected < row_topk) {
        const int logical = indices[(int64_t)query * topk + selected];
        if (logical >= 0 && logical < visible) {
          slot = req_to_token[(int64_t)req_index * req_stride + logical];
        }
      }
      slots[tid] = slot;
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < kLoadIters; ++it) {
      const int vector = tid + it * kThreads;
      const int element_off = vector * kVectorElems;
      const int tok_local = element_off / kDim;
      const int d_off = element_off % kDim;
      const int slot = slots[tok_local];
      const bool valid = tok_local < tile_tokens && slot >= 0;
      const int64_t base = (int64_t)slot * kDim + d_off;
      const int dst = tok_local * kKVStride + d_off;
      load_cache_vector(k_cache, base, ks + dst, valid);
      load_cache_vector(v_cache, base, vs + dst, valid);
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
      const __half2 *qr = reinterpret_cast<const __half2 *>(qs + r * kDim);
      const int n = lane;
      const __half2 *krow =
          reinterpret_cast<const __half2 *>(ks + n * kKVStride);
      float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
#pragma unroll
      for (int h = 0; h < kDim / 8; ++h) {
        const __half2 qh0 = qr[4 * h];
        const __half2 kh0 = krow[4 * h];
        const __half2 qh1 = qr[4 * h + 1];
        const __half2 kh1 = krow[4 * h + 1];
        const __half2 qh2 = qr[4 * h + 2];
        const __half2 kh2 = krow[4 * h + 2];
        const __half2 qh3 = qr[4 * h + 3];
        const __half2 kh3 = krow[4 * h + 3];
        acc0 = fmaf(__half2float(qh0.x), __half2float(kh0.x), acc0);
        acc0 = fmaf(__half2float(qh0.y), __half2float(kh0.y), acc0);
        acc1 = fmaf(__half2float(qh1.x), __half2float(kh1.x), acc1);
        acc1 = fmaf(__half2float(qh1.y), __half2float(kh1.y), acc1);
        acc2 = fmaf(__half2float(qh2.x), __half2float(kh2.x), acc2);
        acc2 = fmaf(__half2float(qh2.y), __half2float(kh2.y), acc2);
        acc3 = fmaf(__half2float(qh3.x), __half2float(kh3.x), acc3);
        acc3 = fmaf(__half2float(qh3.y), __half2float(kh3.y), acc3);
      }
      scores[r * kBlockN + n] = (n < tile_tokens && slots[n] >= 0)
                                    ? acc0 + acc1 + acc2 + acc3
                                    : -1.0e30f;
      __syncwarp();

      float loc_max = scores[r * kBlockN + lane];
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_max = fmaxf(loc_max, __shfl_xor_sync(0xffffffffu, loc_max, off));
      }
      const float m_new = fmaxf(m_row[r], loc_max);
      const float alpha = exp2f((m_row[r] - m_new) * scale_log2);
      m_row[r] = m_new;
      l_row[r] *= alpha;
      if (alpha != 1.f) {
#pragma unroll
        for (int j = 0; j < kAccPerLane; ++j) {
          o_acc[r][j] *= alpha;
        }
      }
      float p = 0.f;
      if (lane < tile_tokens && slots[lane] >= 0) {
        p = exp2f((scores[r * kBlockN + lane] - m_new) * scale_log2);
      }
      probs[r * kBlockN + lane] = __float2half(p);
      float loc_sum = p;
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        loc_sum += __shfl_xor_sync(0xffffffffu, loc_sum, off);
      }
      l_row[r] += loc_sum;
    }
    __syncthreads();

    if (is_compute_warp) {
      const int r = warp;
#pragma unroll
      for (int n = 0; n < kBlockN; ++n) {
        const float p = __half2float(probs[r * kBlockN + n]);
        if (p == 0.f) {
          continue;
        }
        const __half2 *vrow =
            reinterpret_cast<const __half2 *>(vs + n * kKVStride);
#pragma unroll
        for (int j = 0; j < kPairsPerLane; ++j) {
          const __half2 value = vrow[lane + j * 32];
          o_acc[r][2 * j] = fmaf(p, __half2float(value.x), o_acc[r][2 * j]);
          o_acc[r][2 * j + 1] =
              fmaf(p, __half2float(value.y), o_acc[r][2 * j + 1]);
        }
      }
    }
    __syncthreads();
  }

  if (is_compute_warp) {
    const int r = warp;
    const float inv_l = l_row[r] > 0.f ? 1.f / l_row[r] : 0.f;
    __half *out_row = output + ((int64_t)query * kGroup + r) * kDim;
#pragma unroll
    for (int j = 0; j < kPairsPerLane; ++j) {
      out_row[2 * (lane + j * 32)] = __float2half_rn(o_acc[r][2 * j] * inv_l);
      out_row[2 * (lane + j * 32) + 1] =
          __float2half_rn(o_acc[r][2 * j + 1] * inv_l);
    }
  }
}

} // namespace sm70_longctx

void sm70_qsa_decode(torch::Tensor q, torch::Tensor k_cache,
                     torch::Tensor v_cache, torch::Tensor req_to_token,
                     torch::Tensor req_indices, torch::Tensor indices,
                     torch::Tensor seq_lens, int64_t max_splits,
                     int64_t min_tokens_per_split, double softmax_scale,
                     torch::Tensor partial_o, torch::Tensor partial_lse) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  (q.size(1) == 3 || q.size(1) == 6) && q.size(2) == kDim,
              "sm70_qsa_decode expects FP16 Q [batch,3|6,256]");
  TORCH_CHECK((k_cache.scalar_type() == torch::kHalf ||
               k_cache.scalar_type() == torch::kUInt8) &&
                  v_cache.scalar_type() == k_cache.scalar_type() &&
                  k_cache.dim() == 3 && k_cache.size(1) == 1 &&
                  k_cache.size(2) == kDim && v_cache.sizes() == k_cache.sizes(),
              "sm70_qsa_decode expects FP16 or E5M2 byte KV [pool,1,256]");
  TORCH_CHECK(indices.scalar_type() == torch::kInt && indices.dim() == 2 &&
                  indices.size(0) == q.size(0),
              "sm70_qsa_decode expects int32 indices [batch,topk]");
  TORCH_CHECK(req_to_token.scalar_type() == torch::kInt &&
                  req_to_token.dim() == 2 &&
                  req_indices.scalar_type() == torch::kInt &&
                  req_indices.numel() == q.size(0) &&
                  seq_lens.scalar_type() == torch::kInt &&
                  seq_lens.numel() == q.size(0),
              "sm70_qsa_decode expects one int32 request and length per row");
  TORCH_CHECK(max_splits > 0 && min_tokens_per_split > 0,
              "sm70_qsa_decode split parameters must be positive");
  TORCH_CHECK(partial_o.scalar_type() == torch::kHalf && partial_o.dim() == 4 &&
                  partial_o.size(0) == q.size(0) &&
                  partial_o.size(1) == max_splits &&
                  partial_o.size(2) == q.size(1) && partial_o.size(3) == kDim,
              "sm70_qsa_decode partial output has the wrong shape");
  TORCH_CHECK(partial_lse.scalar_type() == torch::kFloat &&
                  partial_lse.dim() == 3 && partial_lse.size(0) == q.size(0) &&
                  partial_lse.size(1) == max_splits &&
                  partial_lse.size(2) == q.size(1),
              "sm70_qsa_decode partial LSE has the wrong shape");
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(1, (unsigned int)max_splits, (unsigned int)q.size(0));
  auto launch = [&](auto *k_ptr) {
    using CacheType = std::remove_cv_t<std::remove_pointer_t<decltype(k_ptr)>>;
    if (q.size(1) == 3) {
      qsa_decode_partial_kernel<3, CacheType><<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const __half *>(q.data_ptr()), k_ptr,
          reinterpret_cast<const CacheType *>(v_cache.data_ptr()),
          req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
          indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
          (int)req_to_token.size(1), (int)indices.size(1), (int)max_splits,
          (int)min_tokens_per_split, (float)softmax_scale,
          reinterpret_cast<__half *>(partial_o.data_ptr()),
          partial_lse.data_ptr<float>());
    } else {
      qsa_decode_partial_kernel<6, CacheType><<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const __half *>(q.data_ptr()), k_ptr,
          reinterpret_cast<const CacheType *>(v_cache.data_ptr()),
          req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
          indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
          (int)req_to_token.size(1), (int)indices.size(1), (int)max_splits,
          (int)min_tokens_per_split, (float)softmax_scale,
          reinterpret_cast<__half *>(partial_o.data_ptr()),
          partial_lse.data_ptr<float>());
    }
  };
  if (k_cache.scalar_type() == torch::kHalf) {
    launch(reinterpret_cast<const __half *>(k_cache.data_ptr()));
  } else {
    launch(k_cache.data_ptr<uint8_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_qsa_indexer_decode(torch::Tensor q, torch::Tensor k_cache,
                             torch::Tensor page_table,
                             torch::Tensor context_lens, int64_t max_model_len,
                             double score_scale, torch::Tensor logits) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == kIndexerHeads && q.size(2) == kIndexerDim,
              "sm70_qsa_indexer_decode expects FP16 Q [batch,4,128]");
  TORCH_CHECK(k_cache.scalar_type() == torch::kHalf && k_cache.dim() == 4 &&
                  (k_cache.size(1) == 4 || k_cache.size(1) == 16) &&
                  k_cache.size(2) == 1 && k_cache.size(3) == kIndexerDim,
              "sm70_qsa_indexer_decode expects FP16 K [pages,4|16,1,128]");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt &&
                  page_table.dim() == 2 && page_table.size(0) == q.size(0) &&
                  context_lens.scalar_type() == torch::kInt &&
                  context_lens.numel() == q.size(0),
              "sm70_qsa_indexer_decode expects int32 page metadata");
  TORCH_CHECK(max_model_len > 0 && score_scale > 0,
              "sm70_qsa_indexer_decode dimensions and scale must be positive");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat && logits.dim() == 2 &&
                  logits.size(0) == q.size(0) &&
                  logits.size(1) == max_model_len,
              "sm70_qsa_indexer_decode logits have the wrong shape");
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((unsigned int)((max_model_len + kIndexerKeysPerBlock - 1) /
                                 kIndexerKeysPerBlock),
                  (unsigned int)q.size(0));
  // Mainline uses full page 64 / compression 4 = index page 16.
  // Retain page 4 for the original fork layout and standalone checks.
  if (k_cache.size(1) == 4) {
    qsa_indexer_decode_kernel<4><<<grid, kIndexerThreads, 0, stream>>>(
        reinterpret_cast<const __half *>(q.data_ptr()),
        reinterpret_cast<const __half *>(k_cache.data_ptr()),
        page_table.data_ptr<int>(), context_lens.data_ptr<int>(),
        (int)page_table.size(1), (int)max_model_len, (float)score_scale,
        logits.data_ptr<float>());
  } else {
    qsa_indexer_decode_kernel<16><<<grid, kIndexerThreads, 0, stream>>>(
        reinterpret_cast<const __half *>(q.data_ptr()),
        reinterpret_cast<const __half *>(k_cache.data_ptr()),
        page_table.data_ptr<int>(), context_lens.data_ptr<int>(),
        (int)page_table.size(1), (int)max_model_len, (float)score_scale,
        logits.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm70_qsa_prefill(torch::Tensor q, torch::Tensor k_cache,
                      torch::Tensor v_cache, torch::Tensor req_to_token,
                      torch::Tensor req_indices, torch::Tensor indices,
                      torch::Tensor seq_lens, double softmax_scale,
                      torch::Tensor output) {
  using namespace sm70_longctx;
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  (q.size(1) == 3 || q.size(1) == 6) && q.size(2) == kDim,
              "sm70_qsa_prefill expects FP16 Q [tokens,3|6,256]");
  TORCH_CHECK((k_cache.scalar_type() == torch::kHalf ||
               k_cache.scalar_type() == torch::kUInt8) &&
                  v_cache.scalar_type() == k_cache.scalar_type() &&
                  k_cache.dim() == 3 && k_cache.size(1) == 1 &&
                  k_cache.size(2) == kDim && v_cache.sizes() == k_cache.sizes(),
              "sm70_qsa_prefill expects FP16 or E5M2 byte KV [pool,1,256]");
  TORCH_CHECK(indices.scalar_type() == torch::kInt && indices.dim() == 2 &&
                  indices.size(0) == q.size(0),
              "sm70_qsa_prefill expects int32 indices [tokens,topk]");
  TORCH_CHECK(
      req_to_token.scalar_type() == torch::kInt && req_to_token.dim() == 2 &&
          req_indices.scalar_type() == torch::kInt &&
          req_indices.numel() == 1 && seq_lens.scalar_type() == torch::kInt &&
          seq_lens.numel() == 1,
      "sm70_qsa_prefill currently supports one int32 request row");
  TORCH_CHECK(output.scalar_type() == torch::kHalf &&
                  output.sizes() == q.sizes(),
              "sm70_qsa_prefill output must match Q");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto *k_ptr) {
    using CacheType = std::remove_cv_t<std::remove_pointer_t<decltype(k_ptr)>>;
    if (q.size(1) == 3) {
      qsa_prefill_kernel<3, CacheType><<<q.size(0), kThreads, 0, stream>>>(
          reinterpret_cast<const __half *>(q.data_ptr()), k_ptr,
          reinterpret_cast<const CacheType *>(v_cache.data_ptr()),
          req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
          indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
          (int)req_to_token.size(1), (int)indices.size(1), (float)softmax_scale,
          reinterpret_cast<__half *>(output.data_ptr()));
    } else {
      qsa_prefill_kernel<6, CacheType><<<q.size(0), kThreads, 0, stream>>>(
          reinterpret_cast<const __half *>(q.data_ptr()), k_ptr,
          reinterpret_cast<const CacheType *>(v_cache.data_ptr()),
          req_to_token.data_ptr<int>(), req_indices.data_ptr<int>(),
          indices.data_ptr<int>(), seq_lens.data_ptr<int>(),
          (int)req_to_token.size(1), (int)indices.size(1), (float)softmax_scale,
          reinterpret_cast<__half *>(output.data_ptr()));
    }
  };
  if (k_cache.scalar_type() == torch::kHalf) {
    launch(reinterpret_cast<const __half *>(k_cache.data_ptr()));
  } else {
    launch(k_cache.data_ptr<uint8_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sm70_qsa_decode", &sm70_qsa_decode,
        "QSA grouped split-KV decode (D256 G3/G6 FP16/E5M2)");
  m.def("sm70_qsa_indexer_decode", &sm70_qsa_indexer_decode,
        "QSA indexer decode scoring (H4 D128 FP16)");
  m.def("sm70_qsa_prefill", &sm70_qsa_prefill,
        "QSA chunk-prefill (D256 G3/G6 FP16/E5M2)");
}
