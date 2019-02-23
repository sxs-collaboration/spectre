// Distributed under the MIT License.
// See LICENSE.txt for details

#pragma once

#include <cstdlib>
#include <memory>
#include <sharp_cxx.h>

#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"

namespace Spectral {
namespace Swsh {
namespace detail {

constexpr size_t coefficients_maximum_l_max = collocation_maximum_l_max;

struct DestroySharpAlm {
  void operator()(sharp_alm_info* to_delete) noexcept {
    sharp_destroy_alm_info(to_delete);
  }
};
// The Coefficients class acts largely as a memory-safe container for a
// `sharp_alm_info*`, required for use of libsharp transform utilities.
// The libsharp utilities are currently constructed to only provide user
// functions with collocation data for spin-weighted functions and
// derivatives. If and when the libsharp utilities are expanded to provide
// spin-weighted coefficients as output, this class should be expanded to
// provide information about the value and storage ordering of those
// coefficients to user code. This should be implemented as an iterator, as is
// done in SwshCollocation.hpp.
class Coefficients {
 public:
  explicit Coefficients(size_t l_max) noexcept;

  ~Coefficients() = default;
  Coefficients() = default;
  Coefficients(const Coefficients&) = delete;
  Coefficients(Coefficients&&) = default;
  Coefficients& operator=(const Coefficients&) = delete;
  Coefficients& operator=(Coefficients&&) = default;
  sharp_alm_info* get_sharp_alm_info() const noexcept {
    return alm_info_.get();
  }

  size_t l_max() const noexcept { return l_max_; }

 private:
  std::unique_ptr<sharp_alm_info, DestroySharpAlm> alm_info_;
  size_t l_max_ = 0;
};

// Function for obtaining a `Coefficients`, which is a thin wrapper around
// the libsharp `alm_info`, needed to perform transformations and iterate over
// coefficients. A lazy static cache is used to avoid repeated computation. See
// the similar implementation in `SwshCollocation.hpp` for details about the
// caching mechanism.
const Coefficients& precomputed_coefficients(size_t l_max) noexcept;
}  // namespace detail
}  // namespace Swsh
}  // namespace Spectral
