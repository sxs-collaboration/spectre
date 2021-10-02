// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"              // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

namespace {
void densitized_stress(
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*> result,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& one_over_w_squared,
    const Scalar<DataVector>& pressure_star,
    const Scalar<DataVector>& h_rho_w_squared_plus_b_squared,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  *result = inv_spatial_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) *= get(pressure_star);
      result->get(i, j) +=
          get(h_rho_w_squared_plus_b_squared) * spatial_velocity.get(i) *
              spatial_velocity.get(j) -
          get(magnetic_field_dot_spatial_velocity) *
              (magnetic_field.get(i) * spatial_velocity.get(j) +
               magnetic_field.get(j) * spatial_velocity.get(i)) -
          magnetic_field.get(i) * magnetic_field.get(j) *
              get(one_over_w_squared);
      result->get(i, j) *= get(sqrt_det_spatial_metric);
    }
  }
}

struct MagneticFieldOneForm : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};
struct TildeSUp : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};
struct DensitizedStress : db::SimpleTag {
  using type = tnsr::II<DataVector, 3, Frame::Inertial>;
};
struct OneOverLorentzFactorSquared : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct PressureStar : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct EnthalpyTimesDensityWSquaredPlusBSquared : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

namespace grmhd::ValenciaDivClean {
namespace detail {
void sources_impl(
    const gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        source_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> source_tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_s_up,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        densitized_stress,
    const gsl::not_null<Scalar<DataVector>*> h_rho_w_squared_plus_b_squared,

    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& magnetic_field_squared,
    const Scalar<DataVector>& one_over_w_squared,
    const Scalar<DataVector>& pressure_star,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        trace_spatial_christoffel_second,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const double constraint_damping_parameter) {
  get(*h_rho_w_squared_plus_b_squared) =
      get(magnetic_field_squared) + get(rest_mass_density) *
                                        get(specific_enthalpy) *
                                        square(get(lorentz_factor));
  ::densitized_stress(densitized_stress, inv_spatial_metric,
                      magnetic_field_dot_spatial_velocity, one_over_w_squared,
                      pressure_star, *h_rho_w_squared_plus_b_squared,
                      spatial_velocity, magnetic_field,
                      sqrt_det_spatial_metric);
  raise_or_lower_index(tilde_s_up, tilde_s, inv_spatial_metric);

  get(*source_tilde_tau) = get(lapse) * get<0, 0>(extrinsic_curvature) *
                               get<0, 0>(*densitized_stress) -
                           get<0>(*tilde_s_up) * get<0>(d_lapse);
  for (size_t m = 1; m < 3; ++m) {
    get(*source_tilde_tau) +=
        get(lapse) *
            (extrinsic_curvature.get(m, 0) * densitized_stress->get(m, 0) +
             extrinsic_curvature.get(0, m) * densitized_stress->get(0, m)) -
        tilde_s_up->get(m) * d_lapse.get(m);
    for (size_t n = 1; n < 3; ++n) {
      get(*source_tilde_tau) += get(lapse) * extrinsic_curvature.get(m, n) *
                                densitized_stress->get(m, n);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    source_tilde_s->get(i) = -(get(tilde_d) + get(tilde_tau)) * d_lapse.get(i);
    for (size_t m = 0; m < 3; ++m) {
      source_tilde_s->get(i) += tilde_s.get(m) * d_shift.get(i, m);
      for (size_t n = 0; n < 3; ++n) {
        source_tilde_s->get(i) += 0.5 * get(lapse) *
                                  d_spatial_metric.get(i, m, n) *
                                  densitized_stress->get(m, n);
      }
    }
  }

  raise_or_lower_index(source_tilde_b, d_lapse, inv_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    source_tilde_b->get(i) *= get(tilde_phi);
    source_tilde_b->get(i) -=
        get(lapse) * get(tilde_phi) * trace_spatial_christoffel_second.get(i);
    for (size_t m = 0; m < 3; ++m) {
      source_tilde_b->get(i) -= tilde_b.get(m) * d_shift.get(m, i);
    }
  }

  trace(source_tilde_phi, extrinsic_curvature, inv_spatial_metric);
  get(*source_tilde_phi) += constraint_damping_parameter;
  get(*source_tilde_phi) *= -1.0 * get(lapse) * get(tilde_phi);
  for (size_t m = 0; m < 3; ++m) {
    get(*source_tilde_phi) += tilde_b.get(m) * d_lapse.get(m);
  }
}
}  // namespace detail

void ComputeSources::apply(
    const gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        source_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> source_tilde_phi,
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
    const double constraint_damping_parameter) {
  Variables<tmpl::list<
      TildeSUp, DensitizedStress, MagneticFieldOneForm,
      hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
      hydro::Tags::MagneticFieldSquared<DataVector>,
      OneOverLorentzFactorSquared, PressureStar,
      EnthalpyTimesDensityWSquaredPlusBSquared,
      gr::Tags::SpatialChristoffelFirstKind<3, Frame::Inertial, DataVector>,
      gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
      gr::Tags::TraceSpatialChristoffelSecondKind<3, Frame::Inertial,
                                                  DataVector>>>
      temp_tensors(get<0>(tilde_s).size());

  auto& magnetic_field_oneform = get<MagneticFieldOneForm>(temp_tensors);
  raise_or_lower_index(make_not_null(&magnetic_field_oneform), magnetic_field,
                       spatial_metric);

  auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&magnetic_field_squared), magnetic_field,
              magnetic_field_oneform);

  auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
          temp_tensors);
  dot_product(make_not_null(&magnetic_field_dot_spatial_velocity),
              magnetic_field_oneform, spatial_velocity);

  auto& one_over_w_squared = get<OneOverLorentzFactorSquared>(temp_tensors);
  get(one_over_w_squared) = 1.0 / square(get(lorentz_factor));

  auto& pressure_star = get<PressureStar>(temp_tensors);
  get(pressure_star) =
      get(pressure) + 0.5 * square(get(magnetic_field_dot_spatial_velocity)) +
      0.5 * get(magnetic_field_squared) * get(one_over_w_squared);

  auto& spatial_christoffel_first_kind = get<
      gr::Tags::SpatialChristoffelFirstKind<3, Frame::Inertial, DataVector>>(
      temp_tensors);
  gr::christoffel_first_kind(make_not_null(&spatial_christoffel_first_kind),
                             d_spatial_metric);
  auto& spatial_christoffel_second_kind = get<
      gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>>(
      temp_tensors);
  raise_or_lower_first_index(make_not_null(&spatial_christoffel_second_kind),
                             spatial_christoffel_first_kind,
                             inv_spatial_metric);
  auto& trace_spatial_christoffel_second =
      get<gr::Tags::TraceSpatialChristoffelSecondKind<3, Frame::Inertial,
                                                      DataVector>>(
          temp_tensors);
  trace_last_indices(make_not_null(&trace_spatial_christoffel_second),
                     spatial_christoffel_second_kind, inv_spatial_metric);

  detail::sources_impl(
      source_tilde_tau, source_tilde_s, source_tilde_b, source_tilde_phi,

      make_not_null(&get<TildeSUp>(temp_tensors)),
      make_not_null(&get<DensitizedStress>(temp_tensors)),
      make_not_null(
          &get<EnthalpyTimesDensityWSquaredPlusBSquared>(temp_tensors)),

      magnetic_field_dot_spatial_velocity, magnetic_field_squared,
      one_over_w_squared, pressure_star, trace_spatial_christoffel_second,

      tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
      sqrt_det_spatial_metric, inv_spatial_metric, d_lapse, d_shift,
      d_spatial_metric, spatial_velocity, lorentz_factor, magnetic_field,

      rest_mass_density, specific_enthalpy, extrinsic_curvature,
      constraint_damping_parameter);
}
}  // namespace grmhd::ValenciaDivClean
