// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/iterator/zip_iterator.hpp>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
namespace TestHelpers {

template <typename... Structure>
Tensor<ComplexModalVector, Structure...> tensor_to_goldberg_coefficients(
    const Tensor<DataVector, Structure...>& nodal_data,
    const size_t l_max) noexcept {
  Tensor<ComplexModalVector, Structure...> goldberg_modal_data{
      square(l_max + 1)};
  SpinWeighted<ComplexDataVector, 0> transform_buffer{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  for (size_t i = 0; i < nodal_data.size(); ++i) {
    transform_buffer.data() = std::complex<double>(1.0, 0.0) * nodal_data[i];
    goldberg_modal_data[i] =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, transform_buffer), l_max)
            .data();
  }
  return goldberg_modal_data;
}

template <typename... Structure>
Tensor<ComplexModalVector, Structure...> tensor_to_libsharp_coefficients(
    const Tensor<DataVector, Structure...>& nodal_data,
    const size_t l_max) noexcept {
  Tensor<ComplexModalVector, Structure...> libsharp_modal_data{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  SpinWeighted<ComplexDataVector, 0> transform_buffer{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  for (size_t i = 0; i < nodal_data.size(); ++i) {
    transform_buffer.data() = std::complex<double>(1.0, 0.0) * nodal_data[i];
    libsharp_modal_data[i] =
        Spectral::Swsh::swsh_transform(l_max, 1, transform_buffer).data();
  }
  return libsharp_modal_data;
}
}  // namespace TestHelpers
}  // namespace Cce
