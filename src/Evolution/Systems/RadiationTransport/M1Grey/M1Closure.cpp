// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
/// \cond
struct MomentumUp {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};
struct MomentumSquared {
  using type = Scalar<DataVector>;
};
/// \endcond

/// Minerbo (maximum entropy) closure for the M1 scheme
double minerbo_closure_function(const double zeta) noexcept {
  return 1.0 / 3.0 +
         square(zeta) * (0.4 - 2.0 / 15.0 * zeta + 0.4 * square(zeta));
}
double minerbo_closure_deriv(const double zeta) noexcept {
  return 0.4 * zeta * (2.0 - zeta + 4.0 * square(zeta));
}
}  // namespace

namespace RadiationTransport {
namespace M1Grey {

void M1Closure(
    const gsl::not_null<Scalar<DataVector>*> closure_factor,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        pressure_tensor,
    const gsl::not_null<Scalar<DataVector>*> comoving_energy_density,
    const gsl::not_null<Scalar<DataVector>*> comoving_momentum_density_normal,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        comoving_momentum_density_spatial,
    const Scalar<DataVector>& energy_density,
    const tnsr::i<DataVector, 3, Frame::Inertial>& momentum_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& fluid_velocity,
    const Scalar<DataVector>& fluid_lorentz_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
        inv_spatial_metric) noexcept {
  // Small number used to avoid divisions by zero
  static constexpr double avoid_divisions_by_zero = 1.e-150;
  // Below small_velocity, we use the v=0 closure,
  // and we do not differentiate between fluid/inertial frames
  static constexpr double small_velocity = 1.e-15;
  // Dimension of spatial tensors
  constexpr size_t spatial_dim = 3;
  // Number of significant digits used in the rootfinding rooting
  // used to find the closure factor
  constexpr size_t root_find_number_of_digits = 6;
  Variables<tmpl::list<
      hydro::Tags::LorentzFactorSquared<DataVector>, MomentumSquared,
      MomentumUp,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
      hydro::Tags::SpatialVelocitySquared<DataVector>>>
      temp_closure_tensors(get(energy_density).size());

  // The main calculation needed for the M1 closure is to find the
  // roots of J^2 zeta^2 = H^a H_a, with J the fluid-frame energy density
  // and H^a the fluid-frame momentum density (0th and 1st moments).

  // We first compute various fluid quantities and contractions
  // with inertial moments needed even if v^2 is small
  auto& w_sqr =
      get<hydro::Tags::LorentzFactorSquared<DataVector>>(temp_closure_tensors);
  get(w_sqr) = square(get(fluid_lorentz_factor));
  auto& v_sqr = get<hydro::Tags::SpatialVelocitySquared<DataVector>>(
      temp_closure_tensors);
  get(v_sqr) = 1. - 1. / get(w_sqr);
  // S^i, the neutrino momentum tensor
  auto& s_M = get<MomentumUp>(temp_closure_tensors);
  raise_or_lower_index(make_not_null(&s_M), momentum_density,
                       inv_spatial_metric);
  // S^i S_i
  auto& s_sqr = get<MomentumSquared>(temp_closure_tensors);
  dot_product(make_not_null(&s_sqr), s_M, momentum_density);
  // v_i, the spatial velocity one-form of the fluid
  auto& v_m =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>(
          temp_closure_tensors);
  raise_or_lower_index(make_not_null(&v_m), fluid_velocity, spatial_metric);

  // Allocate memory for H^i
  tnsr::I<double, 3, Frame::Inertial> H_M(0.);
  // Loop over points
  for (size_t s = 0; s < v_sqr.size(); ++s) {
    const double& v_sqr_pt = get(v_sqr)[s];
    const double& w_sqr_pt = get(w_sqr)[s];
    const double& e_pt = get(energy_density)[s];
    const double& s_sqr_pt = std::max(get(s_sqr)[s], avoid_divisions_by_zero);
    // Ignore complicated closure calculations
    // if the fluid velocity is very small
    if (v_sqr_pt < small_velocity) {
      // Minerbo closure assuming v=0 (see definition of
      // minerbo_closure_function)
      const double zeta = sqrt(s_sqr_pt) / e_pt;
      const double chi = minerbo_closure_function(zeta);
      const double d_thin_e_pt_over_s_sqr = (1.5 * chi - 0.5) * e_pt / s_sqr_pt;
      const double d_thick_e_pt_over_3 = (0.5 - 0.5 * chi) * e_pt;

      get(*closure_factor)[s] = zeta;
      get(*comoving_energy_density)[s] = e_pt;
      get(*comoving_momentum_density_normal)[s] = 0.;
      for (size_t i = 0; i < spatial_dim; i++) {
        comoving_momentum_density_spatial->get(i)[s] =
            momentum_density.get(i)[s];
        // Closure assuming that fluid-frame = inertial frame
        for (size_t j = i; j < spatial_dim; j++) {
          pressure_tensor->get(i, j)[s] =
              d_thick_e_pt_over_3 * inv_spatial_metric.get(i, j)[s] +
              d_thin_e_pt_over_s_sqr * momentum_density.get(i)[s] *
                  momentum_density.get(j)[s];
        }
      }
    }
    // If the fluid velocity cannot be ignored, we need to
    // go through a more expensive closure calculation
    else {
      // Compute useful intermediate quantities from v^i, F_i
      double v_dot_f_pt = 0.;
      for (size_t m = 0; m < spatial_dim; m++) {
        v_dot_f_pt += fluid_velocity.get(m)[s] * momentum_density.get(m)[s];
      }
      // Decomposition of the fluid-frame energy density:
      // J = J0 + d_thin * JThin + d_thick * JThick
      // with d_thin, d_thick=1-d_thin coefficients
      // obtained from the M1 closure.
      const double j_0 = w_sqr_pt * (e_pt - 2. * v_dot_f_pt);
      const double j_thin = w_sqr_pt * e_pt * square(v_dot_f_pt) / s_sqr_pt;
      const double j_thick =
          (w_sqr_pt - 1.) / (1. + 2. * w_sqr_pt) *
          (4. * w_sqr_pt * v_dot_f_pt + e_pt * (3. - 2. * w_sqr_pt));
      // Decomposition of the fluid-frame momentum density:
      // H_a = -( h0T + d_thick hThickT + d_thin hThinT) t_a
      //  - ( h0V + d_thick hThickV + d_thin hThinV) v_a
      //  - ( h0F + d_thick hThickF + d_thin hThinF) F_a
      // with t_a the unit normal, v_a the 3-velocity, and F_a the
      // inertial frame momentum density. This is a decomposition of
      // convenience, which is not unique: F_a and v_a are not
      // orthogonal vectors, but both are normal to t_a.
      const double& w_pt = get(fluid_lorentz_factor)[s];
      const double h_0_t = w_pt * (j_0 + v_dot_f_pt - e_pt);
      const double h_0_v = w_pt * j_0;
      const double h_0_f = -w_pt;
      const double h_thin_t = w_pt * j_thin;
      const double h_thin_v = h_thin_t;
      const double h_thin_f = w_pt * e_pt * v_dot_f_pt / s_sqr_pt;
      const double h_thick_t = w_pt * j_thick;
      const double h_thick_v =
          h_thick_t +
          w_pt / (2. * w_sqr_pt + 1.) *
              ((3. - 2. * w_sqr_pt) * e_pt + (2. * w_sqr_pt - 1.) * v_dot_f_pt);
      const double h_thick_f = w_pt * v_sqr_pt;
      // Quantities needed for the computation of H^2 = H^a H_a,
      // independent of zeta. We write:
      // H^2 = h_sqr_0 + h_sqr_thin * d_thin + h_sqr_thick*d_thick
      // + h_sqr_thin_thin * d_thin^2 + h_sqr_thick_thick * d_thick^2
      // + h_sqr_thin_thick * d_thin * d_thick;
      const double h_sqr_0 = -square(h_0_t) + square(h_0_v) * v_sqr_pt +
                             square(h_0_f) * s_sqr_pt +
                             2. * h_0_v * h_0_f * v_dot_f_pt;
      const double h_sqr_thin =
          2. * (h_0_v * h_thin_v * v_sqr_pt + h_0_f * h_thin_f * s_sqr_pt +
                h_0_v * h_thin_f * v_dot_f_pt + h_0_f * h_thin_v * v_dot_f_pt -
                h_0_t * h_thin_t);
      const double h_sqr_thick =
          2. * (h_0_v * h_thick_v * v_sqr_pt + h_0_f * h_thick_f * s_sqr_pt +
                h_0_v * h_thick_f * v_dot_f_pt +
                h_0_f * h_thick_v * v_dot_f_pt - h_0_t * h_thick_t);
      const double h_sqr_thin_thick =
          2. *
          (h_thin_v * h_thick_v * v_sqr_pt + h_thin_f * h_thick_f * s_sqr_pt +
           h_thin_v * h_thick_f * v_dot_f_pt +
           h_thin_f * h_thick_v * v_dot_f_pt - h_thin_t * h_thick_t);
      const double h_sqr_thick_thick =
          square(h_thick_v) * v_sqr_pt + square(h_thick_f) * s_sqr_pt +
          2. * h_thick_v * h_thick_f * v_dot_f_pt - square(h_thick_t);
      const double h_sqr_thin_thin =
          square(h_thin_v) * v_sqr_pt + square(h_thin_f) * s_sqr_pt +
          2. * h_thin_v * h_thin_f * v_dot_f_pt - square(h_thin_t);

      // Root finding function
      const auto zeta_j_sqr_minus_h_sqr = [
        &e_pt, &j_0, &j_thin, &j_thick, &h_sqr_0, &h_sqr_thick, &h_sqr_thin,
        &h_sqr_thin_thin, &h_sqr_thick_thick, &h_sqr_thin_thick
      ](const double local_zeta) noexcept {
        const double chi = minerbo_closure_function(local_zeta);
        const double dchi_dzeta = minerbo_closure_deriv(local_zeta);
        const double d_thin = 1.5 * chi - 0.5;
        const double d_thick = 1. - d_thin;
        const double d_thin_dzeta = 1.5 * dchi_dzeta;
        const double d_thick_dzeta = -d_thin_dzeta;

        const double e_fluid = j_0 + j_thin * d_thin + j_thick * d_thick;
        const double de_fluid_dzeta =
            j_thin * d_thin_dzeta + j_thick * d_thick_dzeta;
        const double h_sqr = h_sqr_0 + h_sqr_thick * d_thick +
                             h_sqr_thin * d_thin +
                             h_sqr_thin_thin * square(d_thin) +
                             h_sqr_thick_thick * square(d_thick) +
                             h_sqr_thin_thick * d_thin * d_thick;
        const double dh_sqr_dd_thin = h_sqr_thin + h_sqr_thin_thick * d_thick +
                                      2. * h_sqr_thin_thin * d_thin;
        const double dh_sqr_dd_thick = h_sqr_thick + h_sqr_thin_thick * d_thin +
                                       2. * h_sqr_thick_thick * d_thick;
        return std::make_pair(
            (square(e_fluid * local_zeta) - h_sqr) / square(e_pt),
            (2. * e_fluid * de_fluid_dzeta * square(local_zeta) +
             2. * square(e_fluid) * local_zeta - d_thin_dzeta * dh_sqr_dd_thin -
             d_thin_dzeta * dh_sqr_dd_thick) /
                square(e_pt));
      };
      const double& zeta = get(*closure_factor)[s];
      // Initial guess for root finding.
      // Choice 1: previous value
      // Choice 2: value for zero-velocity
      const double zeta_guess =
          (zeta <= avoid_divisions_by_zero or zeta >= 1. ? sqrt(s_sqr_pt) / e_pt
                                                         : zeta);
      get(*closure_factor)[s] =
          RootFinder::newton_raphson(zeta_j_sqr_minus_h_sqr, zeta_guess, 0., 1.,
                                     root_find_number_of_digits);

      // Assemble output quantities:
      const double chi = minerbo_closure_function(zeta);
      const double d_thin = 1.5 * chi - 0.5;
      const double d_thick = 1. - d_thin;
      get(*comoving_energy_density)[s] =
          j_0 + j_thin * d_thin + j_thick * d_thick;
      get(*comoving_momentum_density_normal)[s] =
          h_0_t + h_thin_t * d_thin + h_thick_t * d_thick;
      for (size_t i = 0; i < spatial_dim; i++) {
        comoving_momentum_density_spatial->get(i) =
            -(h_0_v + h_thin_v * d_thin + h_thick_v * d_thick) * v_m.get(i)[s] -
            (h_0_f + h_thin_f * d_thin + h_thick_f * d_thick) *
                momentum_density.get(i)[s];
        for (size_t j = i; j < spatial_dim; j++) {
          // Optically thin part of pressure tensor
          pressure_tensor->get(i, j) = d_thin * e_pt *
                                       momentum_density.get(i)[s] *
                                       momentum_density.get(j)[s] / s_sqr_pt;
        }
      }
      // Optically thick limit
      for (size_t i = 0; i < spatial_dim; i++) {
        H_M.get(i) =
            s_M.get(i)[s] / w_pt +
            fluid_velocity.get(i)[s] * w_pt / (2. * w_sqr_pt + 1.) *
                ((4. * w_sqr_pt + 1.) * v_dot_f_pt - 4. * w_sqr_pt * e_pt);
      }
      const double J_over_3 =
          1. / (2. * w_sqr_pt + 1.) *
          ((2. * w_sqr_pt - 1.) * e_pt - 2. * w_sqr_pt * v_dot_f_pt);
      for (size_t i = 0; i < spatial_dim; i++) {
        for (size_t j = i; j < spatial_dim; j++) {
          pressure_tensor->get(i, j) +=
              d_thick * (J_over_3 * (4. * w_sqr_pt * fluid_velocity.get(i)[s] *
                                         fluid_velocity.get(j)[s] +
                                     inv_spatial_metric.get(i, j)[s]) +
                         w_pt * (H_M.get(i) * fluid_velocity.get(j)[s] +
                                 H_M.get(j) * fluid_velocity.get(i)[s]));
        }
      }
    }
  }
}

}  // namespace M1Grey
}  // namespace RadiationTransport
