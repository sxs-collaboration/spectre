// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

namespace intrp {

std::complex<double> SpanInterpolator::interpolate(
    const gsl::span<const double>& source_points,
    const gsl::span<const std::complex<double>>& values,
    const double target_point) const {
  // the operation below to get the real and imag parts does not alter the
  // contents of the span, so the const-cast is safe.
  const ComplexDataVector view{
      const_cast<std::complex<double>*>(values.data()),  // NOLINT
      values.size()};
  const DataVector real_part = real(view);
  const DataVector imag_part = imag(view);
  return std::complex<double>{
      interpolate(source_points,
                  gsl::span<const double>(real_part.data(), real_part.size()),
                  target_point),
      interpolate(source_points,
                  gsl::span<const double>(imag_part.data(), imag_part.size()),
                  target_point)};
}
}  // namespace intrp
