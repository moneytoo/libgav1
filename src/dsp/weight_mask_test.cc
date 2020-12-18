// Copyright 2020 The libgav1 Authors
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

#include "src/dsp/weight_mask.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"
#include "src/utils/cpu.h"
#include "src/utils/memory.h"
#include "tests/third_party/libvpx/acm_random.h"
#include "tests/utils.h"

namespace libgav1 {
namespace dsp {
namespace {

constexpr int kNumSpeedTests = 50000;
constexpr int kMaxPredictionSize = 128;
// weight_mask is only used with kCompoundPredictionTypeDiffWeighted with
// convolve producing the most extreme ranges, see: src/dsp/convolve.cc &
// src/dsp/warp.cc.
constexpr int kPredictionRange[3][2] = {
    {-5132, 9212},
    {3988, 61352},
    {3974, 61559},
};

const char* GetDigest8bpp(int id) {
  static const char* const kDigest[] = {
      "25a1d6d1b3e75213e12800676686703e",
      "b93b38e538dcb072e4b492a781f909ca",
      "50b5e6680ecdaa95c4e95c220abe5bd8",
      "" /*kBlock16x4*/,
      "fdc4a868311d629c99507f728f56d575",
      "b6da56bbefac4ca4edad1b8f68791606",
      "2dbe65f1cfbe37134bf1dbff11c222f2",
      "6d77edaf6fa479a669a6309722d8f352",
      "4f58c12179012ae1cd1c21e01258a39b",
      "9b3e1ce01d886db45d1295878c3b9e00",
      "97b5be2d7bb19a045b3815a972b918b7",
      "5b2cba7e06155bb4e9e281d6668633df",
      "ca6ea9f694ebfc6fc0c9fc4d22d140ec",
      "0efca5b9f6e5c287ff8683c558382987",
      "36941879ee00efb746c45cad08d6559b",
      "6d8ee22d7dd051f391f295c4fdb617d7",
      "e99ada080a5ddf9df50544301d0b6f6e",
      "acd821f8e49d47a0735ed1027f43d64b",

      // mask_is_inverse = true.
      "c9cd4ae74ed092198f812e864cfca8a2",
      "77125e2710c78613283fe279f179b59d",
      "52df4dae64ef06f913351c61d1082e93",
      "" /*kBlock16x4*/,
      "1fa861d6ca726db3b6ac4fa78bdcb465",
      "4f42297f3fb4cfc3afc3b89b30943c32",
      "730fefde2cd8d65ae8ceca7fb1d1e9f3",
      "cc53bf23217146c77797d3c21fac35b8",
      "55be7f6f22c02f43ccced3131c8ba02b",
      "bf1e12cd57424aee4a35969ad72cbdd3",
      "bea31fa1581e19b7819400f417130ec3",
      "fb42a215163ee9e13b9d7db1838caca2",
      "0747f7ab50b564ad30d73381337ed845",
      "74f5bdb72ae505376596c2d91fd67d27",
      "56b5053da761ffbfd856677bbc34e353",
      "15001c7c9b585e19de875ec6926c2451",
      "35d49b7ec45c42b84fdb30f89ace00fb",
      "9fcb7a44be4ce603a95978acf0fb54d7",
  };
  return kDigest[id];
}

#if LIBGAV1_MAX_BITDEPTH >= 10
const char* GetDigest10bpp(int id) {
  static const char* const kDigest[] = {
      "3cba49e84f5ef8c91e4f4b8c264da6a3", "6848aee4d8a773f04251af76def65acf",
      "17174ec7b8a3066df2648c2e18df4c75", "" /*kBlock4x16*/,
      "a5231e091c1c2a3dc24519f5331f59fb", "2fd787b791a45a1c0f54088462a27e3a",
      "5a5d0cc09b275470c5123377e8119705", "83fb0781045a6315a538fa6205d74683",
      "21747a80d989c946d97eced279208d1a", "b4c40c1d62a39133f86acb211d7a77f5",
      "826c23b4064c5178305fb2af45640b3f", "83ff425cf2d0d97404482cae717a3a73",
      "29667f4ca4b82ed89993a13fdfcb5303", "024740a99a8cd72b3996cc6a0e44608a",
      "1b047f39211c9887366663d3558e25a4", "c234fbdc219ff29794bcef5258042392",
      "e541956018c8ffc17290cfb3d4e4ae77", "15e939efaacffd2660a25385fd33414f",

      "7e01e584ebd09eb256280cca0077ddbe", "72b60ce0c123dd074b6bcedc94f854e9",
      "35a0764bf525863dab08927aa1763500", "" /*kBlock4x16*/,
      "aba1198a2e14ebddd7ae60329b3905ca", "bd16795b64437c3437b1eb8e1d53021f",
      "6da2aec8f98f4df9251b731232981b1c", "c69d8cb3b4ef88e3e8981d5ebed6f23d",
      "a2241995d2e7009e2dd3e8e9fc77e7d5", "76e3431f11b139e2caaf52a4b0e70dfb",
      "5332b9645605e8760225888da5df5b92", "8a7f4b75fc7ae23ba8c72cd0d04b8d20",
      "2742bc65b1d23dd423451696e6e2439c", "15c01dcd7e43e5d10c579ec4c9ed6960",
      "5882f79bc6a0ea52415ae8de331fc12f", "ae7616e818cbe5bc2f7e96ee91281a8e",
      "b59dd900801da0753af28a8b90995cf1", "1558b0ef8ee9e18422291b63f8abadf8",
  };
  return kDigest[id];
}
#endif  // LIBGAV1_MAX_BITDEPTH >= 10

struct WeightMaskTestParam {
  WeightMaskTestParam(int width, int height, bool mask_is_inverse)
      : width(width), height(height), mask_is_inverse(mask_is_inverse) {}
  int width;
  int height;
  bool mask_is_inverse;
};

std::ostream& operator<<(std::ostream& os, const WeightMaskTestParam& param) {
  return os << param.width << "x" << param.height
            << ", mask_is_inverse: " << param.mask_is_inverse;
}

template <int bitdepth>
class WeightMaskTest : public ::testing::TestWithParam<WeightMaskTestParam>,
                       public test_utils::MaxAlignedAllocable {
 public:
  WeightMaskTest() = default;
  ~WeightMaskTest() override = default;

  void SetUp() override {
    test_utils::ResetDspTable(bitdepth);
    WeightMaskInit_C();
    const dsp::Dsp* const dsp = dsp::GetDspTable(bitdepth);
    ASSERT_NE(dsp, nullptr);
    const int width_index = FloorLog2(width_) - 3;
    const int height_index = FloorLog2(height_) - 3;
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    const char* const test_case = test_info->test_suite_name();
    if (absl::StartsWith(test_case, "C/")) {
    } else if (absl::StartsWith(test_case, "NEON/")) {
      WeightMaskInit_NEON();
    } else if (absl::StartsWith(test_case, "SSE41/")) {
      WeightMaskInit_SSE4_1();
    }
    func_ = dsp->weight_mask[width_index][height_index][mask_is_inverse_];
  }

 protected:
  void SetInputData(bool use_fixed_values, int value_1, int value_2);
  void Test(int num_runs, bool use_fixed_values, int value_1, int value_2);

 private:
  const int width_ = GetParam().width;
  const int height_ = GetParam().height;
  const bool mask_is_inverse_ = GetParam().mask_is_inverse;
  alignas(
      kMaxAlignment) uint16_t block_1_[kMaxPredictionSize * kMaxPredictionSize];
  alignas(
      kMaxAlignment) uint16_t block_2_[kMaxPredictionSize * kMaxPredictionSize];
  uint8_t mask_[kMaxPredictionSize * kMaxPredictionSize] = {};
  dsp::WeightMaskFunc func_;
};

template <int bitdepth>
void WeightMaskTest<bitdepth>::SetInputData(const bool use_fixed_values,
                                            const int value_1,
                                            const int value_2) {
  if (use_fixed_values) {
    std::fill(block_1_, block_1_ + kMaxPredictionSize * kMaxPredictionSize,
              value_1);
    std::fill(block_2_, block_2_ + kMaxPredictionSize * kMaxPredictionSize,
              value_2);
  } else {
    constexpr int offset = (bitdepth == 8) ? -kPredictionRange[0][0] : 0;
    constexpr int bitdepth_index = (bitdepth - 8) >> 1;
    libvpx_test::ACMRandom rnd(libvpx_test::ACMRandom::DeterministicSeed());
    auto get_value = [&rnd](int min, int max) {
      int value;
      do {
        value = rnd(max + 1);
      } while (value < min);
      return value;
    };
    const int min = kPredictionRange[bitdepth_index][0] + offset;
    const int max = kPredictionRange[bitdepth_index][1] + offset;

    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        block_1_[y * width_ + x] = get_value(min, max) - offset;
        block_2_[y * width_ + x] = get_value(min, max) - offset;
      }
    }
  }
}

BlockSize DimensionsToBlockSize(int width, int height) {
  if (width == 4) {
    if (height == 4) return kBlock4x4;
    if (height == 8) return kBlock4x8;
    if (height == 16) return kBlock4x16;
    return kBlockInvalid;
  }
  if (width == 8) {
    if (height == 4) return kBlock8x4;
    if (height == 8) return kBlock8x8;
    if (height == 16) return kBlock8x16;
    if (height == 32) return kBlock8x32;
    return kBlockInvalid;
  }
  if (width == 16) {
    if (height == 4) return kBlock16x4;
    if (height == 8) return kBlock16x8;
    if (height == 16) return kBlock16x16;
    if (height == 32) return kBlock16x32;
    if (height == 64) return kBlock16x64;
    return kBlockInvalid;
  }
  if (width == 32) {
    if (height == 8) return kBlock32x8;
    if (height == 16) return kBlock32x16;
    if (height == 32) return kBlock32x32;
    if (height == 64) return kBlock32x64;
    return kBlockInvalid;
  }
  if (width == 64) {
    if (height == 16) return kBlock64x16;
    if (height == 32) return kBlock64x32;
    if (height == 64) return kBlock64x64;
    if (height == 128) return kBlock64x128;
    return kBlockInvalid;
  }
  if (width == 128) {
    if (height == 64) return kBlock128x64;
    if (height == 128) return kBlock128x128;
    return kBlockInvalid;
  }
  return kBlockInvalid;
}

template <int bitdepth>
void WeightMaskTest<bitdepth>::Test(const int num_runs,
                                    const bool use_fixed_values,
                                    const int value_1, const int value_2) {
  if (func_ == nullptr) return;
  SetInputData(use_fixed_values, value_1, value_2);
  const absl::Time start = absl::Now();
  for (int i = 0; i < num_runs; ++i) {
    func_(block_1_, block_2_, mask_, kMaxPredictionSize);
  }
  const absl::Duration elapsed_time = absl::Now() - start;
  if (use_fixed_values) {
    int fixed_value = (value_1 - value_2 == 0) ? 38 : 64;
    if (mask_is_inverse_) fixed_value = 64 - fixed_value;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        ASSERT_EQ(static_cast<int>(mask_[y * kMaxPredictionSize + x]),
                  fixed_value)
            << "x: " << x << " y: " << y;
      }
    }
  } else {
    const int id_offset = mask_is_inverse_ ? kMaxBlockSizes - 4 : 0;
    const int id = id_offset +
                   static_cast<int>(DimensionsToBlockSize(width_, height_)) - 4;
    if (bitdepth == 8) {
      test_utils::CheckMd5Digest(
          absl::StrFormat("BlockSize %dx%d", width_, height_).c_str(),
          "WeightMask", GetDigest8bpp(id), mask_, sizeof(mask_), elapsed_time);
#if LIBGAV1_MAX_BITDEPTH >= 10
    } else {
      test_utils::CheckMd5Digest(
          absl::StrFormat("BlockSize %dx%d", width_, height_).c_str(),
          "WeightMask", GetDigest10bpp(id), mask_, sizeof(mask_), elapsed_time);
#endif
    }
  }
}

const WeightMaskTestParam weight_mask_test_param[] = {
    WeightMaskTestParam(8, 8, false),     WeightMaskTestParam(8, 16, false),
    WeightMaskTestParam(8, 32, false),    WeightMaskTestParam(16, 8, false),
    WeightMaskTestParam(16, 16, false),   WeightMaskTestParam(16, 32, false),
    WeightMaskTestParam(16, 64, false),   WeightMaskTestParam(32, 8, false),
    WeightMaskTestParam(32, 16, false),   WeightMaskTestParam(32, 32, false),
    WeightMaskTestParam(32, 64, false),   WeightMaskTestParam(64, 16, false),
    WeightMaskTestParam(64, 32, false),   WeightMaskTestParam(64, 64, false),
    WeightMaskTestParam(64, 128, false),  WeightMaskTestParam(128, 64, false),
    WeightMaskTestParam(128, 128, false), WeightMaskTestParam(8, 8, true),
    WeightMaskTestParam(8, 16, true),     WeightMaskTestParam(8, 32, true),
    WeightMaskTestParam(16, 8, true),     WeightMaskTestParam(16, 16, true),
    WeightMaskTestParam(16, 32, true),    WeightMaskTestParam(16, 64, true),
    WeightMaskTestParam(32, 8, true),     WeightMaskTestParam(32, 16, true),
    WeightMaskTestParam(32, 32, true),    WeightMaskTestParam(32, 64, true),
    WeightMaskTestParam(64, 16, true),    WeightMaskTestParam(64, 32, true),
    WeightMaskTestParam(64, 64, true),    WeightMaskTestParam(64, 128, true),
    WeightMaskTestParam(128, 64, true),   WeightMaskTestParam(128, 128, true),
};

using WeightMaskTest8bpp = WeightMaskTest<8>;

TEST_P(WeightMaskTest8bpp, FixedValues) {
  const int min = kPredictionRange[0][0];
  const int max = kPredictionRange[0][1];
  Test(1, true, min, min);
  Test(1, true, min, max);
  Test(1, true, max, min);
  Test(1, true, max, max);
}

TEST_P(WeightMaskTest8bpp, RandomValues) { Test(1, false, -1, -1); }

TEST_P(WeightMaskTest8bpp, DISABLED_Speed) {
  Test(kNumSpeedTests, false, -1, -1);
}

INSTANTIATE_TEST_SUITE_P(C, WeightMaskTest8bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#if LIBGAV1_ENABLE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, WeightMaskTest8bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#endif
#if LIBGAV1_ENABLE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE41, WeightMaskTest8bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#endif

#if LIBGAV1_MAX_BITDEPTH >= 10
using WeightMaskTest10bpp = WeightMaskTest<10>;

TEST_P(WeightMaskTest10bpp, FixedValues) {
  const int min = kPredictionRange[1][0];
  const int max = kPredictionRange[1][1];
  Test(1, true, min, min);
  Test(1, true, min, max);
  Test(1, true, max, min);
  Test(1, true, max, max);
}

TEST_P(WeightMaskTest10bpp, RandomValues) { Test(1, false, -1, -1); }

TEST_P(WeightMaskTest10bpp, DISABLED_Speed) {
  Test(kNumSpeedTests, false, -1, -1);
}

INSTANTIATE_TEST_SUITE_P(C, WeightMaskTest10bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#if LIBGAV1_ENABLE_NEON
INSTANTIATE_TEST_SUITE_P(NEON, WeightMaskTest10bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#endif
#if LIBGAV1_ENABLE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE41, WeightMaskTest10bpp,
                         ::testing::ValuesIn(weight_mask_test_param));
#endif
#endif  // LIBGAV1_MAX_BITDEPTH >= 10

}  // namespace
}  // namespace dsp
}  // namespace libgav1
