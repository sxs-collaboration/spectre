// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

/// \ingroup TestingFrameworkGroup
/// Determines if the given `solution` is a time-independent solution
/// of the Einstein equations. Uses numerical derivatives to compute
/// the solution, on a grid extending from `lower_bound` to `upper_bound`
/// with `grid_size_each_dimension` points in each dimension.
/// The right-hand side of the Einstein equations must be zero within
/// `error_tolerance`
template <typename Solution>
void verify_time_independent_einstein_solution(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound,
    const double error_tolerance) noexcept {
  // Shorter names for tags.
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<3, Frame::Inertial>;
  using Pi = ::GeneralizedHarmonic::Tags::Pi<3, Frame::Inertial>;
  using Phi = ::GeneralizedHarmonic::Tags::Phi<3, Frame::Inertial>;
  using GaugeH = ::GeneralizedHarmonic::Tags::GaugeH<3, Frame::Inertial>;
  using VariablesTags = tmpl::list<SpacetimeMetric, Pi, Phi, GaugeH>;

  // Set up grid
  const size_t data_size = pow<3>(grid_size_each_dimension);
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const double t = 1.3;  // Arbitrary time for time-independent solution.

  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& lapse = get<gr::Tags::Lapse<>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<>>>(vars);
  const auto& d_lapse =
      get<typename Solution::template DerivLapse<DataVector>>(vars);
  const auto& shift = get<gr::Tags::Shift<3>>(vars);
  const auto& d_shift =
      get<typename Solution::template DerivShift<DataVector>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<3>>>(vars);
  const auto& g = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& dt_g = get<Tags::dt<gr::Tags::SpatialMetric<3>>>(vars);
  const auto& d_g =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Check those quantities that should vanish identically.
  CHECK(get(dt_lapse) == make_with_value<DataVector>(x, 0.));
  CHECK(dt_shift ==
        make_with_value<cpp20::remove_cvref_t<decltype(dt_shift)>>(x, 0.));
  CHECK(dt_g == make_with_value<cpp20::remove_cvref_t<decltype(dt_g)>>(x, 0.));

  // Need upper spatial metric for many things below.
  const auto upper_spatial_metric = determinant_and_inverse(g).second;

  // Compute generalized harmonic quantities.  Put them in a Variables
  // so that we can take numerical derivatives of them.
  // Also put gauge_function into this list since we need a numerical
  // derivative of it too.
  Variables<VariablesTags> gh_vars(data_size);
  auto& psi = get<SpacetimeMetric>(gh_vars);
  auto& pi = get<Pi>(gh_vars);
  auto& phi = get<Phi>(gh_vars);
  auto& gauge_function = get<GaugeH>(gh_vars);
  psi = gr::spacetime_metric(lapse, shift, g);
  phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift, g, d_g);
  pi = GeneralizedHarmonic::pi(lapse, dt_lapse, shift, dt_shift, g, dt_g, phi);

  // Compute gauge_function compatible with d/dt(lapse) and d/dt(shift).
  gauge_function = GeneralizedHarmonic::gauge_source(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, g,
      trace(gr::extrinsic_curvature(lapse, shift, d_shift, g, dt_g, d_g),
            upper_spatial_metric),
      trace_last_indices(gr::christoffel_first_kind(d_g),
                         upper_spatial_metric));

  // Compute numerical derivatives of psi,pi,phi,H.
  // Normally one should not take numerical derivatives of H for
  // plugging into the RHS of the generalized harmonic equations, but
  // here this is just a test.
  const auto gh_derivs =
      partial_derivatives<VariablesTags, VariablesTags, 3, Frame::Inertial>(
          gh_vars, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_psi =
      get<Tags::deriv<SpacetimeMetric, tmpl::size_t<3>, Frame::Inertial>>(
          gh_derivs);
  const auto& d_pi =
      get<Tags::deriv<Pi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& d_phi =
      get<Tags::deriv<Phi, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);
  const auto& d_H =
      get<Tags::deriv<GaugeH, tmpl::size_t<3>, Frame::Inertial>>(gh_derivs);

  Approx numerical_approx =
      Approx::custom().epsilon(error_tolerance).scale(1.0);

  // Test 3-index constraint
  CHECK_ITERABLE_CUSTOM_APPROX(d_psi, phi, numerical_approx);

  // Compute spacetime deriv of H.
  // Assume time derivative of H is zero, for time-independent solution
  auto d4_H = make_with_value<tnsr::ab<DataVector, 3>>(x, 0.0);
  for (size_t a = 0; a < 4; ++a) {
    for (size_t i = 0; i < 3; ++i) {
      d4_H.get(i + 1, a) = d_H.get(i, a);
    }
  }

  // Compute analytic derivatives of psi, for use in computing
  // christoffel symbols.
  auto d4_psi = make_with_value<tnsr::abb<DataVector, 3>>(x, 0.0);
  for (size_t b = 0; b < 4; ++b) {
    for (size_t c = b; c < 4; ++c) {  // symmetry
      d4_psi.get(0, b, c) = -get(lapse) * pi.get(b, c);
      for (size_t k = 0; k < 3; ++k) {
        d4_psi.get(0, b, c) += shift.get(k) * phi.get(k, b, c);
        d4_psi.get(k + 1, b, c) = phi.get(k, b, c);
      }
    }
  }

  // Compute derived spacetime quantities
  const auto upper_psi =
      gr::inverse_spacetime_metric(lapse, shift, upper_spatial_metric);
  const auto christoffel_first_kind = gr::christoffel_first_kind(d4_psi);
  const auto christoffel_second_kind =
      raise_or_lower_first_index(christoffel_first_kind, upper_psi);
  const auto trace_christoffel_first_kind =
      trace_last_indices(christoffel_first_kind, upper_psi);
  const auto normal_one_form =
      gr::spacetime_normal_one_form<3, Frame::Inertial>(lapse);
  const auto normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Test ADM evolution equation gives zero
  auto dt_g_adm = make_with_value<tnsr::ii<DataVector, 3>>(x, 0.0);
  {
    auto ex_curv = gr::extrinsic_curvature(lapse, shift, d_shift, g, dt_g, d_g);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {  // Symmetry
        dt_g_adm.get(i, j) -= 2.0 * get(lapse) * ex_curv.get(i, j);
        for (size_t k = 0; k < 3; ++k) {
          dt_g_adm.get(i, j) += shift.get(k) * d_g.get(k, i, j) +
                                g.get(k, i) * d_shift.get(j, k) +
                                g.get(k, j) * d_shift.get(i, k);
        }
      }
    }
    CHECK_ITERABLE_APPROX(dt_g_adm,
                          make_with_value<decltype(dt_g_adm)>(x, 0.0));
  }

  // Test 1-index constraint
  auto C_1 = GeneralizedHarmonic::gauge_constraint(
      gauge_function, normal_one_form, normal_vector, upper_spatial_metric,
      upper_psi, pi, phi);
  CHECK_ITERABLE_CUSTOM_APPROX(C_1, make_with_value<decltype(C_1)>(x, 0.0),
                               numerical_approx);

  // Constraint-damping parameters: Set to arbitrary values.
  // gamma = 0 (for all gammas) and gamma1 = -1 are special because,
  // they zero out various terms in the equations, so don't choose those.
  const auto gamma0 = make_with_value<Scalar<DataVector>>(x, 1.0);
  const auto gamma1 = make_with_value<Scalar<DataVector>>(x, 1.0);
  const auto gamma2 = make_with_value<Scalar<DataVector>>(x, 1.0);

  // Compute RHS of generalized harmonic Einstein equations.
  auto dt_psi =
      make_with_value<tnsr::aa<DataVector, 3, Frame::Inertial>>(x, 0.0);
  auto dt_pi =
      make_with_value<tnsr::aa<DataVector, 3, Frame::Inertial>>(x, 0.0);
  auto dt_phi =
      make_with_value<tnsr::iaa<DataVector, 3, Frame::Inertial>>(x, 0.0);
  GeneralizedHarmonic::ComputeDuDt<3>::apply(
      make_not_null(&dt_psi), make_not_null(&dt_pi), make_not_null(&dt_phi),
      psi, pi, phi, d_psi, d_pi, d_phi, gamma0, gamma1, gamma2, gauge_function,
      d4_H, lapse, shift, upper_spatial_metric, upper_psi,
      trace_christoffel_first_kind, christoffel_first_kind,
      christoffel_second_kind, normal_vector, normal_one_form);

  // Make sure the RHS is zero.
  CHECK_ITERABLE_CUSTOM_APPROX(
      dt_psi, make_with_value<decltype(dt_psi)>(x, 0.0), numerical_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_pi, make_with_value<decltype(dt_pi)>(x, 0.0),
                               numerical_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      dt_phi, make_with_value<decltype(dt_phi)>(x, 0.0), numerical_approx);
}
