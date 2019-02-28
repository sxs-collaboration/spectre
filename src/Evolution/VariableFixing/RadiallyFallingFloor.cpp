// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"

#include <algorithm>  // IWYU pragma: keep
#include <pup.h>      // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

namespace VariableFixing {

template <size_t Dim>
RadiallyFallingFloor<Dim>::RadiallyFallingFloor(
    const double minimum_radius_at_which_to_apply_floor,
    const double rest_mass_density_scale, const double rest_mass_density_power,
    const double pressure_scale, const double pressure_power) noexcept
    : minimum_radius_at_which_to_apply_floor_(
          minimum_radius_at_which_to_apply_floor),
      rest_mass_density_scale_(rest_mass_density_scale),
      rest_mass_density_power_(rest_mass_density_power),
      pressure_scale_(pressure_scale),
      pressure_power_(pressure_power) {}

template <size_t Dim>
void RadiallyFallingFloor<Dim>::pup(PUP::er& p) noexcept {  // NOLINT
  p | minimum_radius_at_which_to_apply_floor_;
  p | rest_mass_density_scale_;
  p | rest_mass_density_power_;
  p | pressure_scale_;
  p | pressure_power_;
}

template <size_t Dim>
void RadiallyFallingFloor<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> density,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords) const noexcept {
  const auto radii = magnitude(coords);

  for (size_t i = 0; i < density->get().size(); i++) {
    const double& radius = radii.get()[i];
    if (UNLIKELY(radius < minimum_radius_at_which_to_apply_floor_)) {
      continue;
    }
    pressure->get()[i] = std::max(
        pressure->get()[i], pressure_scale_ * pow(radius, pressure_power_));

    density->get()[i] =
        std::max(density->get()[i], rest_mass_density_scale_ *
                                        pow(radius, rest_mass_density_power_));
  }
}

template <size_t LocalDim>
bool operator==(const RadiallyFallingFloor<LocalDim>& lhs,
                const RadiallyFallingFloor<LocalDim>& rhs) noexcept {
  return lhs.minimum_radius_at_which_to_apply_floor_ ==
             rhs.minimum_radius_at_which_to_apply_floor_ and
         lhs.rest_mass_density_scale_ == rhs.rest_mass_density_scale_ and
         lhs.rest_mass_density_power_ == rhs.rest_mass_density_power_ and
         lhs.pressure_scale_ == rhs.pressure_scale_ and
         lhs.pressure_power_ == rhs.pressure_power_;
}

template <size_t Dim>
bool operator!=(const RadiallyFallingFloor<Dim>& lhs,
                const RadiallyFallingFloor<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                  \
  template class RadiallyFallingFloor<GET_DIM(data)>;           \
  template bool operator==(                                     \
      const RadiallyFallingFloor<GET_DIM(data)>& lhs,           \
      const RadiallyFallingFloor<GET_DIM(data)>& rhs) noexcept; \
  template bool operator!=(                                     \
      const RadiallyFallingFloor<GET_DIM(data)>& lhs,           \
      const RadiallyFallingFloor<GET_DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION

}  // namespace VariableFixing
