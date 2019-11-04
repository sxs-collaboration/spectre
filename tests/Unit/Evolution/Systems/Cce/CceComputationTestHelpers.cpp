// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/Evolution/Systems/Cce/CceComputationTestHelpers.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
namespace TestHelpers {

ComplexDataVector power(const ComplexDataVector& value,
                        const size_t exponent) noexcept {
  ComplexDataVector return_value{value.size(), 1.0};
  for (size_t i = 0; i < exponent; ++i) {
    return_value *= value;
  }
  return return_value;
}

void volume_one_minus_y(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        one_minus_y,
    const size_t l_max) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      get(*one_minus_y).size() / number_of_angular_points;
  const auto& one_minus_y_collocation =
      1.0 - Spectral::collocation_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>(
                number_of_radial_points);
  // iterate through the angular 'chunks' and set them to their 1-y value
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    ComplexDataVector angular_view{
        get(*one_minus_y).data().data() + number_of_angular_points * i,
        number_of_angular_points};
    angular_view = one_minus_y_collocation[i];
  }
}
}  // namespace TestHelpers
}  // namespace Cce
