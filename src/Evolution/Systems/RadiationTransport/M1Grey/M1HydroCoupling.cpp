// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/M1HydroCoupling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct densitized_eta_minus_kappaJ : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct kappaT_lapse : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

namespace RadiationTransport::M1Grey::detail {

void compute_m1_hydro_coupling_impl(
    const gsl::not_null<Scalar<DataVector>*> source_n,
    const gsl::not_null<tnsr::i<DataVector, 3>*> source_i,
    const Scalar<DataVector>& emissivity,
    const Scalar<DataVector>& absorption_opacity,
    const Scalar<DataVector>& scattering_opacity,
    const Scalar<DataVector>& comoving_energy_density,
    const Scalar<DataVector>& comoving_momentum_density_normal,
    const tnsr::i<DataVector, 3>& comoving_momentum_density_spatial,
    const tnsr::I<DataVector, 3>& fluid_velocity,
    const Scalar<DataVector>& fluid_lorentz_factor,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) {
  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       densitized_eta_minus_kappaJ, kappaT_lapse>>
      temp_tensors(get(lapse).size());
  // Dimension of spatial tensors
  constexpr size_t spatial_dim = 3;

  auto& dens_e_minus_kJ = get<densitized_eta_minus_kappaJ>(temp_tensors);
  get(dens_e_minus_kJ) =
      get(lapse) * get(fluid_lorentz_factor) *
      (get(sqrt_det_spatial_metric) * get(emissivity) -
       get(absorption_opacity) * get(comoving_energy_density));
  auto& kT_lapse = get<kappaT_lapse>(temp_tensors);
  get(kT_lapse) =
      get(lapse) * (get(absorption_opacity) + get(scattering_opacity));
  auto& fluid_velocity_i =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&fluid_velocity_i), fluid_velocity,
                       spatial_metric);

  get(*source_n) = get(dens_e_minus_kJ) +
                   get(kT_lapse) * get(comoving_momentum_density_normal);
  for (size_t i = 0; i < spatial_dim; i++) {
    source_i->get(i) = fluid_velocity_i.get(i) * get(dens_e_minus_kJ) -
                       get(kT_lapse) * comoving_momentum_density_spatial.get(i);
  }
}

namespace {
namespace LocalTags {
struct DummySpecies;
using TildeE = Tags::TildeE<Frame::Inertial, DummySpecies>;
using TildeS = Tags::TildeS<Frame::Inertial, DummySpecies>;
using TildeJ = Tags::TildeJ<DummySpecies>;
using TildeHSpatial = Tags::TildeHSpatial<Frame::Inertial, DummySpecies>;
}  // namespace LocalTags
}  // namespace

void compute_m1_hydro_coupling_jacobian_impl(
    const gsl::not_null<Scalar<DataVector>*> deriv_e_source_e,
    const gsl::not_null<tnsr::i<DataVector, 3>*> deriv_e_source_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> deriv_s_source_e,
    const gsl::not_null<tnsr::Ij<DataVector, 3>*> deriv_s_source_s,
    const tnsr::i<DataVector, 3>& tilde_s, const Scalar<DataVector>& tilde_e,
    const Scalar<DataVector>& emissivity,
    const Scalar<DataVector>& absorption_opacity,
    const Scalar<DataVector>& scattering_opacity,
    const tnsr::I<DataVector, 3>& fluid_velocity,
    const Scalar<DataVector>& fluid_lorentz_factor,
    const Scalar<DataVector>& closure_factor,
    const Scalar<DataVector>& comoving_energy_density,
    const tnsr::i<DataVector, 3>& comoving_momentum_density_spatial,
    const Scalar<DataVector>& comoving_momentum_density_normal,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric) {
  const double s_squared_floor = 1.0e-150;

  Variables<tmpl::list<
      ::Tags::TempScalar<0>, ::Tags::TempScalar<1>, ::Tags::Tempi<2, 3>,
      Tags::TildeSVector<Frame::Inertial>, ::Tags::TempI<3, 3>,
      ::Tags::TempScalar<4>, ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
      ::Tags::TempScalar<7>, ::Tags::TempScalar<8>, ::Tags::TempScalar<9>,
      ::Tags::TempScalar<10>, ::Tags::TempScalar<11>, ::Tags::TempScalar<12>,
      ::Tags::TempScalar<13>, ::Tags::TempScalar<14>, ::Tags::Tempi<15, 3>,
      ::Tags::TempScalar<16>, ::Tags::TempScalar<17>, ::Tags::TempScalar<18>,
      ::Tags::TempScalar<19>, ::Tags::TempScalar<20>, ::Tags::TempScalar<21>,
      ::Tags::TempScalar<22>, ::Tags::TempScalar<23>, ::Tags::TempScalar<24>,
      ::Tags::TempScalar<25>, ::Tags::TempI<26, 3>, ::Tags::Tempi<27, 3>,
      ::Tags::TempIj<28, 3>, ::Tags::TempScalar<29>, ::Tags::TempScalar<30>,
      ::Tags::TempI<31, 3>,
      ::imex::Tags::Jacobian<LocalTags::TildeE, LocalTags::TildeJ>,
      ::imex::Tags::Jacobian<LocalTags::TildeS, LocalTags::TildeJ>,
      ::imex::Tags::Jacobian<LocalTags::TildeE, LocalTags::TildeHSpatial>,
      ::imex::Tags::Jacobian<LocalTags::TildeS, LocalTags::TildeHSpatial>,
      ::Tags::TempScalar<32>, ::Tags::TempScalar<33>, ::Tags::TempScalar<34>,
      ::Tags::TempScalar<35>>>
      temporaries(get(emissivity).size());

  auto& d_thin = get<::Tags::TempScalar<0>>(temporaries);
  tenex::evaluate(
      make_not_null(&d_thin),
      0.2 * square(closure_factor()) *
          (3.0 + closure_factor() * (-1.0 + 3.0 * closure_factor())));

  auto& d_thick = get<::Tags::TempScalar<1>>(temporaries);
  tenex::evaluate(make_not_null(&d_thick), 1.0 - d_thin());

  auto& fluid_velocity_lower = get<::Tags::Tempi<2, 3>>(temporaries);
  tenex::evaluate<ti::i>(make_not_null(&fluid_velocity_lower),
                         spatial_metric(ti::i, ti::j) * fluid_velocity(ti::J));

  auto& s_upper = get<Tags::TildeSVector<Frame::Inertial>>(temporaries);
  tenex::evaluate<ti::I>(make_not_null(&s_upper),
                         inverse_spatial_metric(ti::I, ti::J) * tilde_s(ti::j));

  auto& comoving_four_momentum_density_upper =
      get<::Tags::TempI<3, 3>>(temporaries);
  tenex::evaluate<ti::I>(
      make_not_null(&comoving_four_momentum_density_upper),
      comoving_momentum_density_spatial(ti::j) *
              inverse_spatial_metric(ti::I, ti::J) +
          comoving_momentum_density_normal() * fluid_velocity(ti::I));

  auto& inverse_s_norm = get<::Tags::TempScalar<4>>(temporaries);
  tenex::evaluate(make_not_null(&inverse_s_norm),
                  1.0 / (s_upper(ti::I) * tilde_s(ti::i) + s_squared_floor));

  auto& fluid_velocity_norm = get<::Tags::TempScalar<5>>(temporaries);
  tenex::evaluate(make_not_null(&fluid_velocity_norm),
                  fluid_velocity(ti::I) * fluid_velocity_lower(ti::i));

  auto& s_dot_fluid_velocity = get<::Tags::TempScalar<6>>(temporaries);
  tenex::evaluate(make_not_null(&s_dot_fluid_velocity),
                  tilde_s(ti::i) * fluid_velocity(ti::I));

  auto& s_dot_fluid_velocity_normalized =
      get<::Tags::TempScalar<7>>(temporaries);
  tenex::evaluate(make_not_null(&s_dot_fluid_velocity_normalized),
                  s_dot_fluid_velocity() * inverse_s_norm());

  auto& s_dot_fluid_velocity_squared_normalized =
      get<::Tags::TempScalar<8>>(temporaries);
  tenex::evaluate(make_not_null(&s_dot_fluid_velocity_squared_normalized),
                  s_dot_fluid_velocity() * s_dot_fluid_velocity_normalized());

  auto& denom = get<::Tags::TempScalar<9>>(temporaries);
  tenex::evaluate(make_not_null(&denom),
                  1.0 / (1.0 + 2.0 * square(fluid_lorentz_factor())));

  auto& scaled_comoving_energy_density =
      get<::Tags::TempScalar<10>>(temporaries);
  tenex::evaluate(make_not_null(&scaled_comoving_energy_density),
                  square(closure_factor()) * comoving_energy_density());

  auto& h_difference_s_coef = get<::Tags::TempScalar<11>>(temporaries);
  tenex::evaluate(
      make_not_null(&h_difference_s_coef),
      fluid_lorentz_factor() * (fluid_velocity_norm() -
                                tilde_e() * s_dot_fluid_velocity_normalized()));

  auto& common_difference_term = get<::Tags::TempScalar<12>>(temporaries);
  tenex::evaluate(
      make_not_null(&common_difference_term),
      denom() *
          ((2.0 * square(fluid_lorentz_factor()) - 3.0) * tilde_e() -
           4.0 * square(fluid_lorentz_factor()) * s_dot_fluid_velocity()));

  auto& j_difference = get<::Tags::TempScalar<13>>(temporaries);
  tenex::evaluate(make_not_null(&j_difference),
                  square(fluid_lorentz_factor()) *
                      (fluid_velocity_norm() * common_difference_term() +
                       tilde_e() * s_dot_fluid_velocity_squared_normalized()));

  // Negated at use site.
  auto& h_difference_v_coef = get<::Tags::TempScalar<14>>(temporaries);
  tenex::evaluate(
      make_not_null(&h_difference_v_coef),
      fluid_lorentz_factor() *
          (common_difference_term() + j_difference() + s_dot_fluid_velocity()));

  auto& h_difference = get<::Tags::Tempi<15, 3>>(temporaries);
  tenex::evaluate<ti::i>(
      make_not_null(&h_difference),
      h_difference_s_coef() * tilde_s(ti::i) -
          h_difference_v_coef() * fluid_velocity_lower(ti::i));

  auto& deriv_e_h_v_coef = get<::Tags::TempScalar<16>>(temporaries);
  tenex::evaluate(
      make_not_null(&deriv_e_h_v_coef),
      square(fluid_lorentz_factor()) *
          (-4.0 * d_thick() * denom() -
           d_thin() * (1.0 + s_dot_fluid_velocity_squared_normalized())));

  // Negated at use site.
  auto& deriv_e_h_s_coef = get<::Tags::TempScalar<17>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_e_h_s_coef),
                  d_thin() * s_dot_fluid_velocity_normalized());

  auto& deriv_s_h_trace_coef = get<::Tags::TempScalar<18>>(temporaries);
  tenex::evaluate(
      make_not_null(&deriv_s_h_trace_coef),
      (1.0 / fluid_lorentz_factor() + d_thin() * h_difference_s_coef()) /
          fluid_lorentz_factor());

  auto& deriv_s_j_v_coef = get<::Tags::TempScalar<19>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_s_j_v_coef),
                  -2.0 * fluid_lorentz_factor() *
                      (deriv_s_h_trace_coef() +
                       d_thick() * fluid_velocity_norm() * denom()));

  auto& deriv_s_h_vv_coef = get<::Tags::TempScalar<20>>(temporaries);
  tenex::evaluate(
      make_not_null(&deriv_s_h_vv_coef),
      2.0 - denom() * d_thick() +
          2.0 * d_thin() * fluid_lorentz_factor() * h_difference_s_coef());

  // Negated at use site.
  auto& deriv_s_h_vs_coef = get<::Tags::TempScalar<21>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_s_h_vs_coef),
                  d_thin() * tilde_e() * inverse_s_norm());

  auto& deriv_s_h_ss_coef = get<::Tags::TempScalar<22>>(temporaries);
  tenex::evaluate(
      make_not_null(&deriv_s_h_ss_coef),
      2.0 * s_dot_fluid_velocity_normalized() * deriv_s_h_vs_coef());

  auto& deriv_s_j_s_coef = get<::Tags::TempScalar<23>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_s_j_s_coef),
                  -2.0 * fluid_lorentz_factor() *
                      s_dot_fluid_velocity_squared_normalized() *
                      deriv_s_h_vs_coef());

  // Negated at use site.
  auto& deriv_s_h_sv_coef = get<::Tags::TempScalar<24>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_s_h_sv_coef),
                  fluid_lorentz_factor() * deriv_s_j_s_coef());

  auto& constant_d_deriv_e_j_over_lorentz_factor =
      get<::Tags::TempScalar<25>>(temporaries);
  tenex::evaluate(
      make_not_null(&constant_d_deriv_e_j_over_lorentz_factor),
      fluid_lorentz_factor() *
          (d_thin() * (1.0 + s_dot_fluid_velocity_squared_normalized()) +
           3.0 * d_thick() * denom() * (1.0 + fluid_velocity_norm())));

  auto& constant_d_deriv_s_j_over_lorentz_factor =
      get<::Tags::TempI<26, 3>>(temporaries);
  tenex::evaluate<ti::I>(
      make_not_null(&constant_d_deriv_s_j_over_lorentz_factor),
      deriv_s_j_v_coef() * fluid_velocity(ti::I) +
          deriv_s_j_s_coef() * s_upper(ti::I));

  auto& constant_d_deriv_e_h_over_lorentz_factor =
      get<::Tags::Tempi<27, 3>>(temporaries);
  tenex::evaluate<ti::i>(
      make_not_null(&constant_d_deriv_e_h_over_lorentz_factor),
      deriv_e_h_v_coef() * fluid_velocity_lower(ti::i) -
          deriv_e_h_s_coef() * tilde_s(ti::i));

  auto& constant_d_deriv_s_h_over_lorentz_factor =
      get<::Tags::TempIj<28, 3>>(temporaries);
  tenex::evaluate<ti::I, ti::j>(
      make_not_null(&constant_d_deriv_s_h_over_lorentz_factor),
      deriv_s_h_vv_coef() * fluid_velocity(ti::I) *
              fluid_velocity_lower(ti::j) +
          deriv_s_h_ss_coef() * s_upper(ti::I) * tilde_s(ti::j) -
          deriv_s_h_sv_coef() * s_upper(ti::I) * fluid_velocity_lower(ti::j) -
          deriv_s_h_vs_coef() * fluid_velocity(ti::I) * tilde_s(ti::j));
  for (size_t i = 0; i < 3; ++i) {
    constant_d_deriv_s_h_over_lorentz_factor.get(i, i) +=
        get(deriv_s_h_trace_coef);
  }

  auto& deriv_dthin_prefactor = get<::Tags::TempScalar<29>>(temporaries);
  tenex::evaluate(
      make_not_null(&deriv_dthin_prefactor),
      1.0 / (scaled_comoving_energy_density() * j_difference() -
             comoving_four_momentum_density_upper(ti::J) * h_difference(ti::j) +
             5.0 / 3.0 * square(comoving_energy_density()) /
                 (2.0 + closure_factor() * (-1.0 + closure_factor() * 4.0))));

  auto& deriv_e_dthin_over_lorentz_factor =
      get<::Tags::TempScalar<30>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_e_dthin_over_lorentz_factor),
                  deriv_dthin_prefactor() *
                      (comoving_four_momentum_density_upper(ti::J) *
                           constant_d_deriv_e_h_over_lorentz_factor(ti::j) -
                       scaled_comoving_energy_density() *
                           constant_d_deriv_e_j_over_lorentz_factor()));
  auto& deriv_s_dthin_over_lorentz_factor =
      get<::Tags::TempI<31, 3>>(temporaries);
  tenex::evaluate<ti::I>(
      make_not_null(&deriv_s_dthin_over_lorentz_factor),
      deriv_dthin_prefactor() *
          (comoving_four_momentum_density_upper(ti::J) *
               constant_d_deriv_s_h_over_lorentz_factor(ti::I, ti::j) -
           scaled_comoving_energy_density() *
               constant_d_deriv_s_j_over_lorentz_factor(ti::I)));

  auto& deriv_e_j =
      get<::imex::Tags::Jacobian<LocalTags::TildeE, LocalTags::TildeJ>>(
          temporaries);
  tenex::evaluate(make_not_null(&deriv_e_j),
                  fluid_lorentz_factor() *
                      (constant_d_deriv_e_j_over_lorentz_factor() +
                       j_difference() * deriv_e_dthin_over_lorentz_factor()));
  auto& deriv_s_j =
      get<::imex::Tags::Jacobian<LocalTags::TildeS, LocalTags::TildeJ>>(
          temporaries);
  tenex::evaluate<ti::I>(
      make_not_null(&deriv_s_j),
      fluid_lorentz_factor() *
          (constant_d_deriv_s_j_over_lorentz_factor(ti::I) +
           j_difference() * deriv_s_dthin_over_lorentz_factor(ti::I)));
  auto& deriv_e_h =
      get<::imex::Tags::Jacobian<LocalTags::TildeE, LocalTags::TildeHSpatial>>(
          temporaries);
  tenex::evaluate<ti::i>(
      make_not_null(&deriv_e_h),
      fluid_lorentz_factor() *
          (constant_d_deriv_e_h_over_lorentz_factor(ti::i) +
           deriv_e_dthin_over_lorentz_factor() * h_difference(ti::i)));
  auto& deriv_s_h =
      get<::imex::Tags::Jacobian<LocalTags::TildeS, LocalTags::TildeHSpatial>>(
          temporaries);
  tenex::evaluate<ti::I, ti::j>(
      make_not_null(&deriv_s_h),
      fluid_lorentz_factor() *
          (constant_d_deriv_s_h_over_lorentz_factor(ti::I, ti::j) +
           deriv_s_dthin_over_lorentz_factor(ti::I) * h_difference(ti::j)));

  auto& deriv_source_e_j_coef = get<::Tags::TempScalar<32>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_source_e_j_coef),
                  lapse() * fluid_lorentz_factor() * scattering_opacity());
  auto& deriv_source_e_v_coef = get<::Tags::TempScalar<33>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_source_e_v_coef),
                  lapse() * fluid_lorentz_factor() *
                      (absorption_opacity() + scattering_opacity()));
  auto& deriv_source_s_h_coef = get<::Tags::TempScalar<34>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_source_s_h_coef),
                  -lapse() * (absorption_opacity() + scattering_opacity()));
  auto& deriv_source_s_jv_coef = get<::Tags::TempScalar<35>>(temporaries);
  tenex::evaluate(make_not_null(&deriv_source_s_jv_coef),
                  -lapse() * fluid_lorentz_factor() * absorption_opacity());

  tenex::evaluate(deriv_e_source_e, deriv_source_e_j_coef() * deriv_e_j() -
                                        deriv_source_e_v_coef());
  tenex::evaluate<ti::I>(deriv_s_source_e,
                         deriv_source_e_j_coef() * deriv_s_j(ti::I) +
                             deriv_source_e_v_coef() * fluid_velocity(ti::I));
  tenex::evaluate<ti::i>(
      deriv_e_source_s,
      deriv_source_s_h_coef() * deriv_e_h(ti::i) +
          deriv_source_s_jv_coef() * deriv_e_j() * fluid_velocity_lower(ti::i));
  tenex::evaluate<ti::I, ti::j>(
      deriv_s_source_s, deriv_source_s_h_coef() * deriv_s_h(ti::I, ti::j) +
                            deriv_source_s_jv_coef() * deriv_s_j(ti::I) *
                                fluid_velocity_lower(ti::j));
}

}  // namespace RadiationTransport::M1Grey::detail
