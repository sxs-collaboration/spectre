// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/Evolution/Systems/Cce/CceComputationTestHelpers.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
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

void generate_volume_data_from_separated_values(
    const gsl::not_null<ComplexDataVector*> volume_data,
    const gsl::not_null<ComplexDataVector*> one_divided_by_r,
    const ComplexDataVector& angular_collocation,
    const ComplexModalVector& radial_coefficients, const size_t l_max,
    const size_t number_of_radial_grid_points) noexcept {
  for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
    ComplexDataVector volume_angular_view{
        volume_data->data() +
            i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    ComplexDataVector one_divided_by_r_angular_view{
        one_divided_by_r->data() +
            i * Spectral::Swsh::number_of_swsh_collocation_points(l_max),
        Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
    volume_angular_view = angular_collocation * radial_coefficients[0];
    for (size_t radial_power = 1; radial_power < radial_coefficients.size();
         ++radial_power) {
      volume_angular_view += angular_collocation *
                             radial_coefficients[radial_power] *
                             power(one_divided_by_r_angular_view, radial_power);
    }
  }
}
}  // namespace TestHelpers
}  // namespace Cce
