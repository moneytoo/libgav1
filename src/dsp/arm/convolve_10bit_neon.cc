// Copyright 2021 The libgav1 Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/dsp/convolve.h"
#include "src/utils/cpu.h"

#if LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10
#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"
#include "src/utils/compiler_attributes.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace {

void ConvolveCompoundCopy_NEON(
    const void* const reference, const ptrdiff_t reference_stride,
    const int /*horizontal_filter_index*/, const int /*vertical_filter_index*/,
    const int /*horizontal_filter_id*/, const int /*vertical_filter_id*/,
    const int width, const int height, void* const prediction,
    const ptrdiff_t /*pred_stride*/) {
  const auto* src = static_cast<const uint16_t*>(reference);
  const ptrdiff_t src_stride = reference_stride >> 1;
  auto* dest = static_cast<uint16_t*>(prediction);
  constexpr int final_shift =
      kInterRoundBitsVertical - kInterRoundBitsCompoundVertical;
  const uint16x8_t offset =
      vdupq_n_u16((1 << kBitdepth10) + (1 << (kBitdepth10 - 1)));

  if (width >= 16) {
    int y = height;
    do {
      int x = 0;
      int w = width;
      do {
        const uint16x8_t v_src_lo = vld1q_u16(&src[x]);
        const uint16x8_t v_src_hi = vld1q_u16(&src[x + 8]);
        const uint16x8_t v_sum_lo = vaddq_u16(v_src_lo, offset);
        const uint16x8_t v_sum_hi = vaddq_u16(v_src_hi, offset);
        const uint16x8_t v_dest_lo = vshlq_n_u16(v_sum_lo, final_shift);
        const uint16x8_t v_dest_hi = vshlq_n_u16(v_sum_hi, final_shift);
        vst1q_u16(&dest[x], v_dest_lo);
        vst1q_u16(&dest[x + 8], v_dest_hi);
        x += 16;
        w -= 16;
      } while (w != 0);
      src += src_stride;
      dest += width;
    } while (--y != 0);
  } else if (width == 8) {
    int y = height;
    do {
      const uint16x8_t v_src_lo = vld1q_u16(&src[0]);
      const uint16x8_t v_src_hi = vld1q_u16(&src[src_stride]);
      const uint16x8_t v_sum_lo = vaddq_u16(v_src_lo, offset);
      const uint16x8_t v_sum_hi = vaddq_u16(v_src_hi, offset);
      const uint16x8_t v_dest_lo = vshlq_n_u16(v_sum_lo, final_shift);
      const uint16x8_t v_dest_hi = vshlq_n_u16(v_sum_hi, final_shift);
      vst1q_u16(&dest[0], v_dest_lo);
      vst1q_u16(&dest[8], v_dest_hi);
      src += src_stride << 1;
      dest += 16;
      y -= 2;
    } while (y != 0);
  } else {  // width == 4
    int y = height;
    do {
      const uint16x4_t v_src_lo = vld1_u16(&src[0]);
      const uint16x4_t v_src_hi = vld1_u16(&src[src_stride]);
      const uint16x4_t v_sum_lo = vadd_u16(v_src_lo, vget_low_u16(offset));
      const uint16x4_t v_sum_hi = vadd_u16(v_src_hi, vget_low_u16(offset));
      const uint16x4_t v_dest_lo = vshl_n_u16(v_sum_lo, final_shift);
      const uint16x4_t v_dest_hi = vshl_n_u16(v_sum_hi, final_shift);
      vst1_u16(&dest[0], v_dest_lo);
      vst1_u16(&dest[4], v_dest_hi);
      src += src_stride << 1;
      dest += 8;
      y -= 2;
    } while (y != 0);
  }
}

void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth10);
  assert(dsp != nullptr);

  dsp->convolve[0][1][0][0] = ConvolveCompoundCopy_NEON;
}

}  // namespace

void ConvolveInit10bpp_NEON() { Init10bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else   // !(LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10)

namespace libgav1 {
namespace dsp {

void ConvolveInit10bpp_NEON() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_NEON && LIBGAV1_MAX_BITDEPTH >= 10
