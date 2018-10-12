// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"              // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_include <array>

/// \cond
namespace {
tnsr::II<DataVector, 3, Frame::Inertial> densitized_stress(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& pressure) noexcept {
  auto result = inv_spatial_metric;

  const auto magnetic_field_oneform =
      raise_or_lower_index(magnetic_field, spatial_metric);
  const auto magnetic_field_dot_spatial_velocity =
      dot_product(magnetic_field_oneform, spatial_velocity);
  // not const to save an allocation below
  auto magnetic_field_squared =
      dot_product(magnetic_field, magnetic_field_oneform);

  const DataVector one_over_w_squared = 1.0 / square(get(lorentz_factor));
  // p_star = p + p_m = p + b^2/2 = p + ((B^n v_n)^2 + (B^n B_n)/W^2)/2
  const DataVector p_star =
      get(pressure) + 0.5 * (square(get(magnetic_field_dot_spatial_velocity)) +
                             get(magnetic_field_squared) * one_over_w_squared);

  Scalar<DataVector> h_rho_w_squared_plus_b_squared =
      std::move(magnetic_field_squared);
  get(h_rho_w_squared_plus_b_squared) += get(rest_mass_density) *
                                         get(specific_enthalpy) *
                                         square(get(lorentz_factor));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result.get(i, j) *= p_star;
      result.get(i, j) +=
          get(h_rho_w_squared_plus_b_squared) * spatial_velocity.get(i) *
              spatial_velocity.get(j) -
          get(magnetic_field_dot_spatial_velocity) *
              (magnetic_field.get(i) * spatial_velocity.get(j) +
               magnetic_field.get(j) * spatial_velocity.get(i)) -
          magnetic_field.get(i) * magnetic_field.get(j) * one_over_w_squared;
      result.get(i, j) *= get(sqrt_det_spatial_metric);
    }
  }
  return result;
}
}  // namespace

namespace grmhd {
namespace ValenciaDivClean {

void ComputeSources::apply(
    gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> source_tilde_s,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> source_tilde_b,
    gsl::not_null<Scalar<DataVector>*> source_tilde_phi,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& pressure, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const double constraint_damping_parameter) noexcept {
  const auto tilde_s_M = raise_or_lower_index(tilde_s, inv_spatial_metric);
  const auto tilde_s_MN =
      densitized_stress(rest_mass_density, specific_enthalpy, lorentz_factor,
                        spatial_velocity, magnetic_field, spatial_metric,
                        inv_spatial_metric, sqrt_det_spatial_metric, pressure);

  // unroll contributions from m=0 and n=0 to avoid initializing
  // source_tilde_tau to zero
  get(*source_tilde_tau) =
      get(lapse) * get<0, 0>(extrinsic_curvature) * get<0, 0>(tilde_s_MN) -
      get<0>(tilde_s_M) * get<0>(d_lapse);
  for (size_t m = 1; m < 3; ++m) {
    get(*source_tilde_tau) +=
        get(lapse) * (extrinsic_curvature.get(m, 0) * tilde_s_MN.get(m, 0) +
                      extrinsic_curvature.get(0, m) * tilde_s_MN.get(0, m)) -
        tilde_s_M.get(m) * d_lapse.get(m);
    for (size_t n = 1; n < 3; ++n) {
      get(*source_tilde_tau) +=
          get(lapse) * extrinsic_curvature.get(m, n) * tilde_s_MN.get(m, n);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    source_tilde_s->get(i) = -(get(tilde_d) + get(tilde_tau)) * d_lapse.get(i);
    for (size_t m = 0; m < 3; ++m) {
      source_tilde_s->get(i) += tilde_s.get(m) * d_shift.get(i, m);
      for (size_t n = 0; n < 3; ++n) {
        source_tilde_s->get(i) += 0.5 * get(lapse) *
                                  d_spatial_metric.get(i, m, n) *
                                  tilde_s_MN.get(m, n);
      }
    }
  }

  const auto trace_of_christoffel_second_kind = trace_last_indices(
      raise_or_lower_first_index(gr::christoffel_first_kind(d_spatial_metric),
                                 inv_spatial_metric),
      inv_spatial_metric);
  *source_tilde_b = raise_or_lower_index(d_lapse, inv_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    source_tilde_b->get(i) *= get(tilde_phi);
    source_tilde_b->get(i) -=
        get(lapse) * get(tilde_phi) * trace_of_christoffel_second_kind.get(i);
    for (size_t m = 0; m < 3; ++m) {
      source_tilde_b->get(i) -= tilde_b.get(m) * d_shift.get(m, i);
    }
  }

  get(*source_tilde_phi) =
      (-get(trace(extrinsic_curvature, inv_spatial_metric)) -
       constraint_damping_parameter) *
      get(lapse) * get(tilde_phi);
  for (size_t m = 0; m < 3; ++m) {
    get(*source_tilde_phi) += tilde_b.get(m) * d_lapse.get(m);
  }
}
}  // namespace ValenciaDivClean
}  // namespace grmhd
/// \endcond
