// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"              // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

namespace grmhd::ValenciaDivClean {
namespace detail {
void fluxes_impl(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    // Temporaries
    const gsl::not_null<Scalar<DataVector>*> transport_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& pressure_star_lapse_sqrt_det_spatial_metric,

    // Extra args
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity) noexcept {
  DataVector& transport_velocity_I = get(*transport_velocity);
  for (size_t i = 0; i < 3; ++i) {
    transport_velocity_I = get(lapse) * spatial_velocity.get(i) - shift.get(i);
    tilde_d_flux->get(i) = get(tilde_d) * transport_velocity_I;
    tilde_tau_flux->get(i) =
        get(tilde_tau) * transport_velocity_I +
        get(pressure_star_lapse_sqrt_det_spatial_metric) *
            spatial_velocity.get(i) -
        get(lapse) * get(magnetic_field_dot_spatial_velocity) * tilde_b.get(i);
    tilde_phi_flux->get(i) =
        get(lapse) * tilde_b.get(i) - get(tilde_phi) * shift.get(i);
    for (size_t j = 0; j < 3; ++j) {
      tilde_s_flux->get(i, j) = tilde_s.get(j) * transport_velocity_I -
                                lapse_b_over_w.get(j) * tilde_b.get(i);
      tilde_b_flux->get(i, j) =
          tilde_b.get(j) * transport_velocity_I +
          get(lapse) * (get(tilde_phi) * inv_spatial_metric.get(i, j) -
                        spatial_velocity.get(j) * tilde_b.get(i));
    }
    tilde_s_flux->get(i, i) += get(pressure_star_lapse_sqrt_det_spatial_metric);
  }
}
}  // namespace detail

void ComputeFluxes::apply(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field) noexcept {
  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
                       hydro::Tags::MagneticFieldSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      temp_tensors{get<0>(shift).size()};

  auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&spatial_velocity_one_form),
                       spatial_velocity, spatial_metric);
  auto& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&magnetic_field_one_form), magnetic_field,
                       spatial_metric);
  auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
          temp_tensors);
  dot_product(make_not_null(&magnetic_field_dot_spatial_velocity),
              magnetic_field, spatial_velocity_one_form);
  auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&magnetic_field_squared), magnetic_field,
              magnetic_field_one_form);

  DataVector& one_over_w_squared =
      get(get<::Tags::TempScalar<0>>(temp_tensors));
  one_over_w_squared = 1.0 / square(get(lorentz_factor));
  // p_star = p + p_m = p + b^2/2 = p + ((B^m v_m)^2 + (B^m B_m)/W^2)/2
  Scalar<DataVector>& pressure_star_lapse_sqrt_det_spatial_metric =
      get<::Tags::TempScalar<1>>(temp_tensors);
  get(pressure_star_lapse_sqrt_det_spatial_metric) =
      get(sqrt_det_spatial_metric) * get(lapse) *
      (get(pressure) + 0.5 * square(get(magnetic_field_dot_spatial_velocity)) +
       0.5 * get(magnetic_field_squared) * one_over_w_squared);

  // lapse b_i / W = lapse (B_i / W^2 + v_i (B^m v_m)
  tnsr::i<DataVector, 3, Frame::Inertial>& lapse_b_over_w =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  for (size_t i = 0; i < 3; ++i) {
    lapse_b_over_w.get(i) *= get(magnetic_field_dot_spatial_velocity);
    lapse_b_over_w.get(i) +=
        one_over_w_squared * magnetic_field_one_form.get(i);
    lapse_b_over_w.get(i) *= get(lapse);
  }

  // Outside the loop to save allocations
  detail::fluxes_impl(tilde_d_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux,
                      tilde_phi_flux,
                      // Temporaries
                      make_not_null(&get<::Tags::TempScalar<2>>(temp_tensors)),
                      lapse_b_over_w, magnetic_field_dot_spatial_velocity,
                      pressure_star_lapse_sqrt_det_spatial_metric,
                      // Extra args
                      tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                      shift, inv_spatial_metric, spatial_velocity);
}
}  // namespace grmhd::ValenciaDivClean
