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

#include "src/dsp/loop_filter.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <string>

#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "src/dsp/dsp.h"
#include "src/utils/constants.h"
#include "src/utils/cpu.h"
#include "tests/block_utils.h"
#include "tests/third_party/libvpx/acm_random.h"
#include "tests/third_party/libvpx/md5_helper.h"
#include "tests/utils.h"

namespace libgav1 {
namespace dsp {
namespace {

// Horizontal and Vertical need 32x32: 8  pixels preceding filtered section
//                                     16 pixels within filtered section
//                                     8  pixels following filtered section
constexpr int kNumPixels = 1024;
constexpr int kBlockStride = 32;

constexpr int kNumTests = 50000;
constexpr int kNumSpeedTests = 500000;

constexpr int kMaxLoopFilter = 63;

template <typename Pixel>
void InitInput(Pixel* dst, const int stride, const int bitdepth,
               libvpx_test::ACMRandom& rnd, const uint8_t inner_thresh,
               const bool transpose) {
  const int max_pixel = (1 << bitdepth) - 1;
  const int pixel_range = max_pixel + 1;
  Pixel tmp[kNumPixels];
  auto clip_pixel = [max_pixel](int val) {
    return static_cast<Pixel>(std::max(std::min(val, max_pixel), 0));
  };

  for (int i = 0; i < kNumPixels;) {
    const uint8_t val = rnd.Rand8();
    if (val & 0x80) {  // 50% chance to choose a new value.
      tmp[i++] = rnd(pixel_range);
    } else {  // 50% chance to repeat previous value in row X times.
      int j = 0;
      while (j++ < ((val & 0x1f) + 1) && i < kNumPixels) {
        if (i < 1) {
          tmp[i] = rnd(pixel_range);
        } else if (val & 0x20) {  // Increment by a value within the limit.
          tmp[i] = clip_pixel(tmp[i - 1] + (inner_thresh - 1));
        } else {  // Decrement by a value within the limit.
          tmp[i] = clip_pixel(tmp[i - 1] - (inner_thresh - 1));
        }
        ++i;
      }
    }
  }

  for (int i = 0; i < kNumPixels;) {
    const uint8_t val = rnd.Rand8();
    if (val & 0x80) {
      ++i;
    } else {  // 50% chance to repeat previous value in column X times.
      int j = 0;
      while (j++ < ((val & 0x1f) + 1) && i < kNumPixels) {
        if (i < 1) {
          tmp[i] = rnd(pixel_range);
        } else if (val & 0x20) {  // Increment by a value within the limit.
          tmp[(i % 32) * 32 + i / 32] = clip_pixel(
              tmp[((i - 1) % 32) * 32 + (i - 1) / 32] + (inner_thresh - 1));
        } else {  // Decrement by a value within the inner_thresh.
          tmp[(i % 32) * 32 + i / 32] = clip_pixel(
              tmp[((i - 1) % 32) * 32 + (i - 1) / 32] - (inner_thresh - 1));
        }
        ++i;
      }
    }
  }

  for (int i = 0; i < kNumPixels; ++i) {
    const int offset = transpose ? stride * (i % stride) + i / stride : i;
    dst[i] = tmp[offset];
  }
}

template <int bitdepth, typename Pixel>
class LoopFilterTest : public testing::TestWithParam<LoopFilterSize> {
 public:
  LoopFilterTest() = default;
  LoopFilterTest(const LoopFilterTest&) = delete;
  LoopFilterTest& operator=(const LoopFilterTest&) = delete;
  ~LoopFilterTest() override = default;

 protected:
  void SetUp() override {
    test_utils::ResetDspTable(bitdepth);
    LoopFilterInit_C();

    const Dsp* const dsp = GetDspTable(bitdepth);
    ASSERT_NE(dsp, nullptr);
    memcpy(base_loop_filters_, dsp->loop_filters[size_],
           sizeof(base_loop_filters_));

    const testing::TestInfo* const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    const char* const test_case = test_info->test_suite_name();
    if (absl::StartsWith(test_case, "C/")) {
      memset(base_loop_filters_, 0, sizeof(base_loop_filters_));
    } else if (absl::StartsWith(test_case, "SSE41/")) {
      if ((GetCpuInfo() & kSSE4_1) != 0) {
        LoopFilterInit_SSE4_1();
      }
    } else if (absl::StartsWith(test_case, "NEON/")) {
      LoopFilterInit_NEON();
    } else {
      FAIL() << "Unrecognized architecture prefix in test case name: "
             << test_case;
    }

    memcpy(cur_loop_filters_, dsp->loop_filters[size_],
           sizeof(cur_loop_filters_));

    for (int i = 0; i < kNumLoopFilterTypes; ++i) {
      // skip functions that haven't been specialized for this particular
      // architecture.
      if (cur_loop_filters_[i] == base_loop_filters_[i]) {
        cur_loop_filters_[i] = nullptr;
      }
    }
  }

  // Check |digests| if non-NULL otherwise print the filter timing.
  void TestRandomValues(const char* const digests[kNumLoopFilterTypes],
                        int num_runs) const;
  void TestSaturatedValues() const;

  const LoopFilterSize size_ = GetParam();
  LoopFilterFunc base_loop_filters_[kNumLoopFilterTypes];
  LoopFilterFunc cur_loop_filters_[kNumLoopFilterTypes];
};

template <int bitdepth, typename Pixel>
void LoopFilterTest<bitdepth, Pixel>::TestRandomValues(
    const char* const digests[kNumLoopFilterTypes], const int num_runs) const {
  for (int i = 0; i < kNumLoopFilterTypes; ++i) {
    libvpx_test::ACMRandom rnd(libvpx_test::ACMRandom::DeterministicSeed());
    if (cur_loop_filters_[i] == nullptr) continue;

    libvpx_test::MD5 md5_digest;
    absl::Duration elapsed_time;
    for (int n = 0; n < num_runs; ++n) {
      Pixel dst[kNumPixels];
      const auto outer_thresh =
          static_cast<uint8_t>(rnd(3 * kMaxLoopFilter + 5));
      const auto inner_thresh = static_cast<uint8_t>(rnd(kMaxLoopFilter + 1));
      const auto hev_thresh =
          static_cast<uint8_t>(rnd(kMaxLoopFilter + 1) >> 4);
      InitInput(dst, kBlockStride, bitdepth, rnd, inner_thresh, (n & 1) == 0);

      const absl::Time start = absl::Now();
      cur_loop_filters_[i](dst + 8 + kBlockStride * 8, kBlockStride,
                           outer_thresh, inner_thresh, hev_thresh);
      elapsed_time += absl::Now() - start;

      md5_digest.Add(reinterpret_cast<const uint8_t*>(dst), sizeof(dst));
    }
    if (digests == nullptr) {
      const auto elapsed_time_us =
          static_cast<int>(absl::ToInt64Microseconds(elapsed_time));
      printf("Mode %s[%25s]: %5d us\n",
             ToString(static_cast<LoopFilterSize>(size_)),
             ToString(static_cast<LoopFilterType>(i)), elapsed_time_us);
    } else {
      const std::string digest = md5_digest.Get();
      printf("Mode %s[%25s]: MD5: %s\n",
             ToString(static_cast<LoopFilterSize>(size_)),
             ToString(static_cast<LoopFilterType>(i)), digest.c_str());
      EXPECT_STREQ(digests[i], digest.c_str());
    }
  }
}

template <int bitdepth, typename Pixel>
void LoopFilterTest<bitdepth, Pixel>::TestSaturatedValues() const {
  const LoopFilterType filter = kLoopFilterTypeHorizontal;
  if (cur_loop_filters_[filter] == nullptr) return;

  Pixel dst[kNumPixels], ref[kNumPixels];
  const auto value = static_cast<Pixel>((1 << bitdepth) - 1);
  for (auto& r : dst) r = value;
  memcpy(ref, dst, sizeof(dst));

  const int outer_thresh = 24;
  const int inner_thresh = 8;
  const int hev_thresh = 0;
  cur_loop_filters_[filter](dst + 8 + kBlockStride * 8, kBlockStride,
                            outer_thresh, inner_thresh, hev_thresh);
  ASSERT_TRUE(test_utils::CompareBlocks(ref, dst, kBlockStride, kBlockStride,
                                        kBlockStride, kBlockStride, true))
      << "kLoopFilterTypeHorizontal output doesn't match reference";
}

//------------------------------------------------------------------------------

using LoopFilterTest8bpp = LoopFilterTest<8, uint8_t>;

const char* const* GetDigests8bpp(LoopFilterSize size) {
  static const char* const kDigestsSize4[kNumLoopFilterTypes] = {
      "2e07bdb04b363d4ce69c7d738b1ee01a",
      "7ff41f2ffa809a2016d342d92afa7f89",
  };
  static const char* const kDigestsSize6[kNumLoopFilterTypes] = {
      "2cd4d9ee7497ed67e38fad9cbeb7e278",
      "75c57a30a927d1aca1ac5c4f175712ca",
  };
  static const char* const kDigestsSize8[kNumLoopFilterTypes] = {
      "854860a272d58ace223454ea727a6fe4",
      "4129ee49b047777583c0e9b2006c87bf",
  };
  static const char* const kDigestsSize14[kNumLoopFilterTypes] = {
      "6eb768620b7ccc84b6f88b9193b02ad2",
      "56e034d9edbe0d5a3cae69b2d9b3486e",
  };

  switch (size) {
    case kLoopFilterSize4:
      return kDigestsSize4;
    case kLoopFilterSize6:
      return kDigestsSize6;
    case kLoopFilterSize8:
      return kDigestsSize8;
    case kLoopFilterSize14:
      return kDigestsSize14;
    default:
      ADD_FAILURE() << "Unknown loop filter size" << size;
      return nullptr;
  }
}

TEST_P(LoopFilterTest8bpp, DISABLED_Speed) {
  TestRandomValues(nullptr, kNumSpeedTests);
}

TEST_P(LoopFilterTest8bpp, FixedInput) {
  TestRandomValues(GetDigests8bpp(size_), kNumTests);
}

TEST_P(LoopFilterTest8bpp, SaturatedValues) { TestSaturatedValues(); }

constexpr LoopFilterSize kLoopFilterSizes[] = {
    kLoopFilterSize4, kLoopFilterSize6, kLoopFilterSize8, kLoopFilterSize14};

INSTANTIATE_TEST_SUITE_P(C, LoopFilterTest8bpp,
                         testing::ValuesIn(kLoopFilterSizes));

#if LIBGAV1_ENABLE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE41, LoopFilterTest8bpp,
                         testing::ValuesIn(kLoopFilterSizes));
#endif
#if LIBGAV1_ENABLE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, LoopFilterTest8bpp,
                         testing::ValuesIn(kLoopFilterSizes));
#endif
//------------------------------------------------------------------------------

#if LIBGAV1_MAX_BITDEPTH >= 10
using LoopFilterTest10bpp = LoopFilterTest<10, uint16_t>;

const char* const* GetDigests10bpp(LoopFilterSize size) {
  static const char* const kDigestsSize4[kNumLoopFilterTypes] = {
      "657dd0f612734c9c1fb50a2313567af4",
      "b1c0a0a0b35bad1589badf3c291c0461",
  };
  static const char* const kDigestsSize6[kNumLoopFilterTypes] = {
      "d41906d4830157052d5bde417d9df9fc",
      "451490def78bd649d16d64db4e665a62",
  };
  static const char* const kDigestsSize8[kNumLoopFilterTypes] = {
      "a763127680f31db7184f2a63ee140268",
      "1f413bebacaa2435f0e07963a9095243",
  };
  static const char* const kDigestsSize14[kNumLoopFilterTypes] = {
      "f0e61add3e5856657c4055751a6dd6e2",
      "44da25d613ea601bf5f6e2a42d329cf0",
  };

  switch (size) {
    case kLoopFilterSize4:
      return kDigestsSize4;
    case kLoopFilterSize6:
      return kDigestsSize6;
    case kLoopFilterSize8:
      return kDigestsSize8;
    case kLoopFilterSize14:
      return kDigestsSize14;
    default:
      ADD_FAILURE() << "Unknown loop filter size" << size;
      return nullptr;
  }
}

TEST_P(LoopFilterTest10bpp, DISABLED_Speed) {
  TestRandomValues(nullptr, kNumSpeedTests);
}

TEST_P(LoopFilterTest10bpp, FixedInput) {
  TestRandomValues(GetDigests10bpp(size_), kNumTests);
}

INSTANTIATE_TEST_SUITE_P(C, LoopFilterTest10bpp,
                         testing::ValuesIn(kLoopFilterSizes));

#if LIBGAV1_ENABLE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE41, LoopFilterTest10bpp,
                         testing::ValuesIn(kLoopFilterSizes));
#endif
#endif

}  // namespace

static std::ostream& operator<<(std::ostream& os, const LoopFilterSize size) {
  return os << ToString(size);
}

}  // namespace dsp
}  // namespace libgav1
