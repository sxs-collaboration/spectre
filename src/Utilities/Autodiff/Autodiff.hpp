// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <autodiff/common/numbertraits.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/var.hpp>

#include "Utilities/Simd/Simd.hpp"

namespace autodiff::detail {
/// Template specialization for simd::batch<double> to treat it as arithmetic.
template <>
struct ArithmeticTraits<simd::batch<double>> {
  static constexpr bool isArithmetic = true;
};

}  // namespace autodiff::detail
