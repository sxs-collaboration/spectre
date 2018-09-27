// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"              // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_include <array>

/// \cond
namespace grmhd {
namespace ValenciaDivClean {

void ConservativeFromPrimitive::apply(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Scalar<DataVector>& divergence_cleaning_field) noexcept {
  const auto spatial_velocity_one_form =
      raise_or_lower_index(spatial_velocity, spatial_metric);
  const auto magnetic_field_one_form =
      raise_or_lower_index(magnetic_field, spatial_metric);
  const auto magnetic_field_dot_spatial_velocity =
      dot_product(magnetic_field, spatial_velocity_one_form);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity_one_form);
  // not const to save an allocation later
  auto magnetic_field_squared =
      dot_product(magnetic_field, magnetic_field_one_form);

  get(*tilde_d) = get(sqrt_det_spatial_metric) * get(rest_mass_density) *
                  get(lorentz_factor);

  get(*tilde_tau) = get(sqrt_det_spatial_metric) *
                      (square(get(lorentz_factor)) *
                        (get(rest_mass_density) *
                          (get(specific_internal_energy) +
                           get(spatial_velocity_squared) * get(lorentz_factor) /
                              (get(lorentz_factor) + 1.)) +
                         get(pressure) * get(spatial_velocity_squared)) +
                       0.5 * get(magnetic_field_squared) *
                        (1.0 + get(spatial_velocity_squared)) -
                       0.5 * square(get(magnetic_field_dot_spatial_velocity)));

  Scalar<DataVector> common_factor = std::move(magnetic_field_squared);
  get(common_factor) *= get(sqrt_det_spatial_metric);
  get(common_factor) +=
      get(*tilde_d) * get(lorentz_factor) * get(specific_enthalpy);
  for (size_t i = 0; i < 3; ++i) {
    tilde_s->get(i) = get(common_factor) * spatial_velocity_one_form.get(i) -
                      get(magnetic_field_dot_spatial_velocity) *
                          get(sqrt_det_spatial_metric) *
                          magnetic_field_one_form.get(i);
  }
  for (size_t i = 0; i < 3; ++i) {
    tilde_b->get(i) = get(sqrt_det_spatial_metric) * magnetic_field.get(i);
  }
  get(*tilde_phi) =
      get(sqrt_det_spatial_metric) * get(divergence_cleaning_field);
}

}  // namespace ValenciaDivClean
}  // namespace grmhd
/// \endcond
