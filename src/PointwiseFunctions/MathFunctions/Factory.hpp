// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "Utilities/TMPL.hpp"

namespace MathFunctions {
namespace Factory_detail {
template <size_t VolumeDim, typename Fr>
struct all_math_functions {
  using type = tmpl::list<MathFunctions::Gaussian<VolumeDim, Fr>>;
};

template <typename Fr>
struct all_math_functions<1, Fr> {
  using type =
      tmpl::list<MathFunctions::Gaussian<1, Fr>, MathFunctions::PowX<1, Fr>,
                 MathFunctions::Sinusoid<1, Fr>>;
};
}  // namespace Factory_detail

template <size_t VolumeDim, typename Fr>
using all_math_functions =
    typename Factory_detail::all_math_functions<VolumeDim, Fr>::type;
}  // namespace MathFunctions
