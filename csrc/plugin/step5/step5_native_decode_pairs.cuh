// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Paired verification decode ported from vLLM v0.23 (49527a385).
// Included inside step5_native_decode, after the Decode definition.

__device__ __forceinline__ int frag_col(int t0, int i) {
  return 2 * t0 + (i & 1) + ((i >> 2) & 1) * 8;
}
__device__ __forceinline__ int frag_token(int i) { return (i >> 1) & 1; }

__device__ __forceinline__ void copy_paired_region(
    __ppu_bfloat16* dst, const PagedKV& kv, int region, bool value) {
  const bool valid = region >= 0;
  const auto* src = kv.data + kv.offset(valid ? unsigned(region) : 0) +
                    (value ? kHeadDim : 0);
  aiu_copy_group(dst, src, kRegionTokens, valid ? 0 : kRegionTokens,
                 kRegionTokens, kv.row_bytes());
}

constexpr int kDsaPairMetadataThreads = 1024;

__device__ __forceinline__ int step4_dsa_paged_decode_pair_prefix(int x,
                                                                  int* sums,
                                                                  int& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int prefix = x;
#pragma unroll
  for (int d = 1; d < 32; d *= 2) {
    const int v = __shfl_up_sync(0xffffffffu, prefix, d);
    if (lane >= d) prefix += v;
  }
  if (lane == 31) sums[warp] = prefix;
  __syncthreads();
  int warp_prefix = sums[lane];
#pragma unroll
  for (int d = 1; d < 32; d *= 2) {
    const int value = __shfl_up_sync(0xffffffffu, warp_prefix, d);
    if (lane >= d) warp_prefix += value;
  }
  const int before =
      warp == 0 ? 0 : __shfl_sync(0xffffffffu, warp_prefix, warp - 1);
  total = __shfl_sync(0xffffffffu, warp_prefix, 31);
  return before + prefix - x;
}

template <bool PACKED>
__global__ void step4_dsa_paged_decode_pair_metadata(
    Decode p, int32_t* packed, int32_t* counts, int physical_regions) {
#if __HGGC_ARCH__ >= 150
  __shared__ __align__(16) int keys[4096], slots[PACKED ? 1 : 4096];
  __shared__ int common[1024], sums[32];
  constexpr int n = 513, cap = 1026;
  const int pair = blockIdx.x, tid = threadIdx.x;
  const int row0 = (pair / 2) * 4 + (pair & 1), row1 = row0 + 2;
  const int pos0 = p.active[row0] ? checked_position(p.positions[row0]) : -1;
  const int pos1 = p.active[row1] ? checked_position(p.positions[row1]) : -1;
  int id0 = -1, id1 = -1, count0 = 0, count1 = 0;
  if (tid < n) {
    const int64_t logical0 = p.regions[int64_t(row0) * p.rs + tid];
    const int64_t logical1 = p.regions[int64_t(row1) * p.rs + tid];
    if (pos0 >= 0 && logical0 >= 0 && logical0 <= pos0 / 8) {
      id0 = p.kv.physical_region(pair / 2, logical0);
      if (id0 >= 0) count0 = min(8, pos0 - int(logical0) * 8 + 1);
    }
    if (pos1 >= 0 && logical1 >= 0 && logical1 <= pos1 / 8) {
      id1 = p.kv.physical_region(pair / 2, logical1);
      if (id1 >= 0) count1 = min(8, pos1 - int(logical1) * 8 + 1);
    }
  }
  int unmatched = 0;
  reinterpret_cast<int4*>(keys)[tid] = make_int4(-1, -1, -1, -1);
  if constexpr (!PACKED)
    reinterpret_cast<int4*>(slots)[tid] = make_int4(-1, -1, -1, -1);
  common[tid] = 0;
  __syncthreads();
  if (tid < n) {
    const int i = tid;
    const int id = id0;

    if (id >= 0 && id < physical_regions) {
      unsigned pos = (static_cast<unsigned>(id) * 2654435761u) >> 20;
      if constexpr (PACKED) {
        while (atomicCAS(keys + pos, -1, id | (i << 21)) != -1)
          pos = (pos + 1) & 4095;
      } else {
        while (atomicCAS(keys + pos, -1, id) != -1) pos = (pos + 1) & 4095;
        slots[pos] = i;
      }
    }
  }
  __syncthreads();
  if (tid < n) {
    const int i = tid;
    const int id = id1;
    const int logical = count1;
    int matched = -1;
    if (id >= 0 && id < physical_regions) {
      unsigned pos = (static_cast<unsigned>(id) * 2654435761u) >> 20;
      for (int probe = 0; probe < 4096; ++probe) {
        const int k = keys[pos];
        if (k == -1) break;
        if constexpr (PACKED) {
          if ((k & 2097151) == id && k >= 0) {
            const int old = atomicOr(keys + pos, static_cast<int>(0x80000000u));
            if (old >= 0) {
              matched = old >> 21;
              break;
            }
          }
        } else {
          if (k == id) {
            const int index = atomicExch(slots + pos, -1);
            if (index >= 0) {
              matched = index;
              break;
            }
          }
        }
        pos = (pos + 1) & 4095;
      }
    }
    unmatched = matched < 0 ? 1 : 0;
    if (matched >= 0) {
      common[matched] = 16 | logical;
    }
  }
  __syncthreads();
  const int x = (common[tid] != 0) + (unmatched << 16);
  int total = 0;
  const int prefix = step4_dsa_paged_decode_pair_prefix(x, sums, total);
  const int shared_count = total & 65535;
  const int common_prefix = prefix & 65535;
  const int unmatched_prefix = prefix >> 16;
  if (tid == 0) {
    counts[pair * 2 + 1] = shared_count;
    counts[pair * 2] = n + (total >> 16);
  }
  if (tid < n) {
    const int dst =
        common[tid] ? common_prefix : shared_count + tid - common_prefix;
    packed[(pair * 3) * cap + dst] =
        id0;
    packed[(pair * 3 + 1) * cap + dst] = count0;
    packed[(pair * 3 + 2) * cap + dst] = common[tid] & 15;
    if (unmatched) {
      const int dst1 = n + unmatched_prefix;
      packed[(pair * 3) * cap + dst1] =
          id1;
      packed[(pair * 3 + 1) * cap + dst1] = -1;
      packed[(pair * 3 + 2) * cap + dst1] =
          count1;
    }
  }
#endif
}

template <int SPLITS>
__global__
__launch_bounds__(64, 2) void step4_dsa_paged_decode_shared_pairs_kernel(
    Decode p, const int32_t* regions, const int32_t* counts, float* maxima,
    float* denominators, int physical_regions) {
#if __HGGC_ARCH__ >= 150
  const float scale = p.scale;
  int region_k = p.slots;
  constexpr int R = 4;
  constexpr int NT = 2;
  using Stage = __ppu_bfloat16[R][3][8][64];
  __shared__ __align__(128) Stage qstage[2];
  const int warp = __all_sync(0xffffffffu, threadIdx.x >= 32) ? 1 : 0;
  const int lane = threadIdx.x & 31;
  const int head = lane >> 2;
  const int t0 = lane & 3;
  const int lr = lane & 7;
  const int lu = lane >> 3;
  const int pairs = gridDim.x / SPLITS;
  const int task = (blockIdx.x % pairs) * SPLITS + blockIdx.x / pairs;
  const int pair = task / SPLITS;
  const int qt = (pair / 2) * 4 + (pair & 1) + warp * 2;
  const int original_n = region_k;
  const int cap = region_k * 2;
  const int common_n = counts[pair * 2 + 1];
  const int total_n = counts[pair * 2];
  region_k = common_n;
  const int split = task % SPLITS;
  const int tiles = (region_k + R - 1) / R;
  const int begin = (tiles * split / SPLITS) * R;
  const int count = min(region_k, (tiles * (split + 1) / SPLITS) * R) - begin;
  const int64_t rb = static_cast<int64_t>(pair) * 3 * cap;
  MmaOperand qf[12];
  const auto* qp =
      p.q + static_cast<int64_t>(qt) * p.qs;
#pragma unroll
  for (int t = 0; t < 2; ++t)
    aiu_copy_group(&qstage[warp][t][0][0][0], qp, 16, t * 8, 8,
                   p.qh * sizeof(__ppu_bfloat16));
  const int first_slot = begin + lane;
  const int first_id =
      first_slot < region_k && lane < count ? regions[rb + first_slot] : -1;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int id = __shfl_sync(0xffffffffu, first_id, r);
    if (warp == 0 && id >= 0 && id < physical_regions)
      copy_paired_region(&qstage[r >> 1][2 + (r & 1)][0][0][0], p.kv, id, false);
  }
  asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
  asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
#pragma unroll
  for (int k = 0; k < 12; ++k) {
    const int qc = (((k % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
    ldm_x4(qf[k], &qstage[warp][lu & 1][k / 4][lr][qc * 8]);
  }
  MmaAccumulator out[12];
#pragma unroll
  for (int j = 0; j < 12; ++j) clear_fragment(out[j]);
  float maximum[2] = {-INFINITY, -INFINITY};
  float denom[2] = {0, 0};
  int cr = -1, cs = -1;
  __syncthreads();
  for (int b = 0; b < count; b += R) {
    if ((b & 31) == 0) {
      const int slot = begin + b + lane;
      const bool valid = slot < region_k && b + lane < count;
      cr = valid ? regions[rb + slot] : -1;
      cs = valid ? regions[rb + (warp + 1) * cap + slot] : -1;
    }
    int ids[R], logical[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      ids[r] = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
      logical[r] = __shfl_sync(0xffffffffu, cs, (b + r) & 31);
      if (b + r >= count || begin + b + r >= region_k) ids[r] = -1;
    }
    if (b > 0) {
      if (warp == 0)
        asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
      __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int vid = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
      if (warp == 0)
        copy_paired_region(&qstage[r >> 1][r & 1][0][0][0], p.kv,
                           b + r < count && vid >= 0 && vid < physical_regions ? vid : -1, true);
    }
    if (warp == 0) asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    bool active_lane = false, all_active_lane = true;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const bool valid =
          logical[r] > 0 && ids[r] >= 0 && ids[r] < physical_regions;
      active_lane |= valid;
      all_active_lane &= valid;
    }
    const bool active = __any_sync(0xffffffffu, active_lane);
    const bool all_active = __all_sync(0xffffffffu, all_active_lane);
    uint32_t token_valid = 0, nan_mask = 0;
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (logical[r] <= 0) nan_mask |= 1u << r;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int token = t * 16 + frag_col(t0, i);
        const int r = token / 8;
        if (ids[r] >= 0 && ids[r] < physical_regions &&
            token % 8 < logical[r])
          token_valid |= 1u << (t * 8 + i);
      }
    }
    MmaOperand pf[NT];
    float rescale[2];
    if (active) {
      MmaAccumulator scores[NT];
      MmaOperand kf;
#pragma unroll
      for (int t = 0; t < NT; ++t) clear_fragment(scores[t]);
#pragma unroll
      for (int k = 0; k < 12; ++k) {
        const int chunk = (((k % 4) * 16 + 8 * (lu & 1)) >> 3) ^ lr;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          ldm_x4(kf, &qstage[t][2 + (lu >> 1)][k / 4][lr][chunk * 8]);
          mma_bf16(scores[t], qf[k], kf, scores[t]);
        }
      }
      float lm[2] = {-INFINITY, -INFINITY};
      uint32_t valid_bits = 0;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const bool valid = token_valid & (1u << (t * 8 + i));
          if (valid) valid_bits |= 1u << (t * 8 + i);
          const float s = valid ? scores[t].x[i] : -INFINITY;
          scores[t].x[i] = s;
          lm[frag_token(i)] = fmaxf(lm[frag_token(i)], s);
        }
      }
      float next[2];
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        const float tm = head_max(lm[tau]);
        const bool valid = tm > -INFINITY;
        next[tau] = valid ? fmaxf(maximum[tau], tm) : maximum[tau];
        rescale[tau] =
            valid ? __expf((maximum[tau] - next[tau]) * scale) : 1.0f;
      }

      float sum[2] = {0, 0};
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const int tau = frag_token(i);
          const float offset = next[tau] == -INFINITY ? 0.0f : next[tau];
          float x0 = (scores[t].x[i] - offset) * scale;
          float x1 = (scores[t].x[i + 1] - offset) * scale;
          if (!(scale > 0)) {
            if (!(valid_bits & (1u << (t * 8 + i)))) x0 = -INFINITY;
            if (!(valid_bits & (1u << (t * 8 + i + 1)))) x1 = -INFINITY;
          }
          const float p0 = __expf(x0), p1 = __expf(x1);
          const __ppu_bfloat162 packed = __floats2bfloat162_rn(p0, p1);
          reinterpret_cast<uint32_t*>(&pf[t].x[0])[i / 2] =
              *reinterpret_cast<const uint32_t*>(&packed);
          sum[tau] += p0;
          sum[tau] += p1;
        }
      }
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        denom[tau] = denom[tau] * rescale[tau] + sum[tau];
        maximum[tau] = next[tau];
      }
    }
    if (warp == 0) asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
    __syncthreads();
    if (b + R < count) {
      const int slot = begin + b + R + lane;
      const int next_cr =
          ((b + R) & 31) == 0
              ? (slot < region_k && b + R + lane < count ? regions[rb + slot]
                                                         : -1)
              : cr;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int id = __shfl_sync(0xffffffffu, next_cr, (b + R + r) & 31);
        if (warp == 0 && b + R + r < count && id >= 0 &&
            id < physical_regions)
          copy_paired_region(&qstage[r >> 1][2 + (r & 1)][0][0][0], p.kv, id, false);
      }
      if (warp == 0)
        asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    }
    if (active) {
      MmaOperand vf;
#pragma unroll
      for (int j = 0; j < 12; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          out[j].x[i] *= frag_token(i) == 0 ? rescale[0] : rescale[1];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const int chunk = (((j % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
          ldm_x4_trans(vf, &qstage[t][lu & 1][j / 4][lr][chunk * 8]);
          if (!all_active) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int r = t * 2 + (i & 1);
              if (nan_mask & (1u << r))
                reinterpret_cast<uint32_t*>(&vf.x[0])[i] = 0;
            }
          }
          mma_bf16(out[j], pf[t], vf, out[j]);
        }
      }
    }
    if (b + R >= count) __syncthreads();
  }
  {
    Stage& sk = qstage[warp];
    Stage& sv = sk;
    const int private_begin = warp == 0 ? common_n : original_n;
    const int private_end = warp == 0 ? original_n : total_n;
    const int n = private_end - private_begin;
    const int tiles = (n + R - 1) / R;
    const int begin = private_begin + (tiles * split / SPLITS) * R;
    const int count =
        min(private_end, private_begin + (tiles * (split + 1) / SPLITS) * R) -
        begin;
    const int region_k = private_end;
    int cr = -1, cs = -1;
    for (int b = 0; b < count; b += R) {
      if ((b & 31) == 0) {
        const int slot = begin + b + lane;
        const bool valid = slot < region_k && b + lane < count;
        cr = valid ? regions[rb + slot] : -1;
        cs = valid ? regions[rb + (warp + 1) * cap + slot] : -1;
      }
      int ids[R], logical[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        ids[r] = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
        logical[r] = __shfl_sync(0xffffffffu, cs, (b + r) & 31);
        if (b + r >= count || begin + b + r >= region_k) ids[r] = -1;
        const bool valid = ids[r] >= 0 && ids[r] < physical_regions && logical[r] > 0;
        copy_paired_region(&sk[r][0][0][0], p.kv, valid ? ids[r] : -1, false);
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
      asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
      MmaAccumulator scores[NT];
      MmaOperand kf;
#pragma unroll
      for (int t = 0; t < NT; ++t) clear_fragment(scores[t]);
#pragma unroll
      for (int k = 0; k < 12; ++k) {
        const int chunk = (((k % 4) * 16 + 8 * (lu & 1)) >> 3) ^ lr;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          ldm_x4(kf, &sk[t * 2 + (lu >> 1)][k / 4][lr][chunk * 8]);
          mma_bf16(scores[t], qf[k], kf, scores[t]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const bool valid =
            ids[r] >= 0 && ids[r] < physical_regions && logical[r] > 0;
        copy_paired_region(&sv[r][0][0][0], p.kv, valid ? ids[r] : -1, true);
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
      float lm[2] = {-INFINITY, -INFINITY};
      uint32_t valid_bits = 0;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int token = t * 16 + frag_col(t0, i);
          const int r = token / 8;
          const bool valid = ids[r] >= 0 && ids[r] < physical_regions &&
                             token % 8 < logical[r];
          if (valid) valid_bits |= 1u << (t * 8 + i);
          const float s = valid ? scores[t].x[i] : -INFINITY;
          scores[t].x[i] = s;
          lm[frag_token(i)] = fmaxf(lm[frag_token(i)], s);
        }
      }
      float next[2], rescale[2];
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        const float tm = head_max(lm[tau]);
        const bool valid = tm > -INFINITY;
        next[tau] = valid ? fmaxf(maximum[tau], tm) : maximum[tau];
        rescale[tau] =
            valid ? __expf((maximum[tau] - next[tau]) * scale) : 1.0f;
      }
      MmaOperand pf[NT];
      float sum[2] = {0, 0};
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const int tau = frag_token(i);
          const float offset = next[tau] == -INFINITY ? 0.0f : next[tau];
          float x0 = (scores[t].x[i] - offset) * scale;
          float x1 = (scores[t].x[i + 1] - offset) * scale;
          if (!(scale > 0)) {
            if (!(valid_bits & (1u << (t * 8 + i)))) x0 = -INFINITY;
            if (!(valid_bits & (1u << (t * 8 + i + 1)))) x1 = -INFINITY;
          }
          const float p0 = __expf(x0), p1 = __expf(x1);
          const __ppu_bfloat162 packed = __floats2bfloat162_rn(p0, p1);
          reinterpret_cast<uint32_t*>(&pf[t].x[0])[i / 2] =
              *reinterpret_cast<const uint32_t*>(&packed);
          sum[tau] += p0;
          sum[tau] += p1;
        }
      }
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        denom[tau] = denom[tau] * rescale[tau] + sum[tau];
        maximum[tau] = next[tau];
      }
      asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
#pragma unroll
      for (int j = 0; j < 12; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          out[j].x[i] *= frag_token(i) == 0 ? rescale[0] : rescale[1];
      }
      MmaOperand vf;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int j = 0; j < 12; ++j) {
          const int chunk = (((j % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
          ldm_x4_trans(vf, &sv[t * 2 + (lu & 1)][j / 4][lr][chunk * 8]);
          mma_bf16(out[j], pf[t], vf, out[j]);
        }
      }
    }
  }
#pragma unroll
  for (int tau = 0; tau < 2; ++tau) {
    const float d = head_sum(denom[tau]);
    const int64_t scalar =
        (static_cast<int64_t>(qt) * 16 + head + tau * 8) *
            SPLITS +
        split;
    if (t0 == 0) {
      maxima[scalar] = d > 0 ? maximum[tau] * scale : -INFINITY;
      denominators[scalar] = d;
    }
#pragma unroll
    for (int j = 0; j < 12; ++j) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = tau * 2 + half * 4;
        const int64_t dst =
            (((static_cast<int64_t>(qt) * SPLITS + split) * 12 + j) * 4 +
             tau * 2 + half) *
                32 +
            lane;
        __stcg(reinterpret_cast<float2*>(p.partial) + dst,
               make_float2(out[j].x[i], out[j].x[i + 1]));
      }
    }
  }
#endif
}

template <int SPLITS>
__global__ void step4_dsa_paged_decode_merge_native(Decode p, const float* split_max,
                                               const float* split_denom) {
#if __HGGC_ARCH__ >= 150
  const int qt = blockIdx.x / 12, j = blockIdx.x % 12;
  if (!p.active[qt] && !p.clear_inactive) return;
  const int tid = threadIdx.x & 63;
  const int tau = tid / 32, half = (tid & 31) / 16;
  const int head = qt * 16 + (tid & 15) / 2 + tau * 8;
  const int col = j * 16 + half * 8 + (tid & 1) * 4;
  float ms[SPLITS], ds[SPLITS];
  float4 xs[SPLITS];
#pragma unroll
  for (int s = 0; s < SPLITS; ++s) {
    ds[s] = split_denom[head * SPLITS + s];
    ms[s] = ds[s] > 0 ? split_max[head * SPLITS + s] : -INFINITY;
    xs[s] = ds[s] > 0 ? __ldcg(reinterpret_cast<const float4*>(p.partial) +
                               ((qt * SPLITS + s) * 12 + j) * 64 + tid)
                      : make_float4(0, 0, 0, 0);
  }
  float m = -INFINITY, denom = 0.0f;
#pragma unroll
  for (int s = 0; s < SPLITS; ++s) m = fmaxf(m, ms[s]);
  float4 acc = make_float4(0, 0, 0, 0);
#pragma unroll
  for (int s = 0; s < SPLITS; ++s) {
    const float weight = ds[s] > 0 ? __expf(ms[s] - m) : 0.0f;
    denom += ds[s] * weight;
    acc.x += xs[s].x * weight;
    acc.y += xs[s].y * weight;
    acc.z += xs[s].z * weight;
    acc.w += xs[s].w * weight;
  }
  const float inv = denom > 0 ? 1.0f / denom : 0.0f;
  const int local_head = head % 16;
  const float gate = p.gate_value(qt, local_head);
  auto* dst = reinterpret_cast<__ppu_bfloat162*>(
      p.out + int64_t(qt) * p.os + local_head * p.oh + col);
  dst[0] = p.output_pair(acc.x * inv, acc.y * inv, gate);
  dst[1] = p.output_pair(acc.z * inv, acc.w * inv, gate);
#endif
}

void launch_paired(Decode p, const at::TensorOptions& options, hggcStream_t stream) {
  constexpr int splits = 8, slots = 513;
  const int pairs = p.rows / 2;
  const int64_t physical_regions = int64_t(p.kv.page_count) << p.kv.region_shift;
  const auto ints = options.dtype(at::kInt);
  auto pair_metadata = at::empty({pairs, 3, 2 * slots}, ints);
  auto pair_counts = at::empty({pairs, 2}, ints);
  auto stats = at::empty({2, p.rows, 16, splits}, options);
  auto acc = at::empty({p.rows, splits, 16, kHeadDim}, options);
  p.partial = acc.data_ptr<float>();
  auto* maxima = stats.data_ptr<float>();
  auto* denominators = maxima + p.rows * 16 * splits;
  auto* metadata = pair_metadata.data_ptr<int32_t>();
  auto* counts = pair_counts.data_ptr<int32_t>();
  if (physical_regions <= (1 << 21)) {
    step4_dsa_paged_decode_pair_metadata<true>
        <<<pairs, kDsaPairMetadataThreads, 0, stream>>>(p, metadata, counts, physical_regions);
  } else {
    step4_dsa_paged_decode_pair_metadata<false>
        <<<pairs, kDsaPairMetadataThreads, 0, stream>>>(p, metadata, counts, physical_regions);
  }
  step4_dsa_paged_decode_shared_pairs_kernel<splits>
      <<<pairs * splits, 64, 0, stream>>>(
          p, metadata, counts, maxima, denominators, physical_regions);
  step4_dsa_paged_decode_merge_native<splits>
      <<<p.rows * 12, 64, 0, stream>>>(p, maxima, denominators);
}
