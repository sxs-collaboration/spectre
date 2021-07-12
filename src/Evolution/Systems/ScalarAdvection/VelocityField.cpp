// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/VelocityField.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarAdvection::Tags {

template <size_t Dim>
void VelocityFieldCompute<Dim>::function(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity_field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) noexcept {
  destructive_resize_components(velocity_field,
                                get_size(get<0>(inertial_coords)));
  if constexpr (Dim == 1) {
    // 1D : advection to +x direction with velocity 1.0
    get<0>(*velocity_field) = 1.0;
  } else if constexpr (Dim == 2) {
    // 2D : rotation about (0.5, 0.5)
    for (size_t i = 0; i < get_size(get<0>(inertial_coords)); ++i) {
      const auto& x = get<0>(inertial_coords)[i];
      const auto& y = get<1>(inertial_coords)[i];
      get<0>(*velocity_field)[i] = 0.5 - y;
      get<1>(*velocity_field)[i] = -0.5 + x;
    }
  }
}

}  // namespace ScalarAdvection::Tags

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                         \
  template void                                                      \
  ScalarAdvection::Tags::VelocityFieldCompute<DIM(data)>::function(  \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*> velocity_field, \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&         \
          inertial_coords) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2))

#undef DIM
#undef INSTANTIATE
