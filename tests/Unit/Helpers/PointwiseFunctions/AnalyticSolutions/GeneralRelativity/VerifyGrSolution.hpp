// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/DuDtTempTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Tags::deriv
/// \endcond

namespace TestHelpers {
/// \ingroup TestingFrameworkGroup
/// Functions for testing GR analytic solutions
namespace VerifyGrSolution {

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
    const std::array<double, 3>& upper_bound, const double error_tolerance) {
  static_assert(is_analytic_solution_v<Solution>,
                "Solution was not derived from AnalyticSolution");
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
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
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
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars);
  const auto& shift = get<gr::Tags::Shift<3>>(vars);
  const auto& d_shift =
      get<::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<3>>>(vars);
  const auto& g = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& dt_g = get<Tags::dt<gr::Tags::SpatialMetric<3>>>(vars);
  const auto& d_g =
      get<::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, Frame::Inertial>>(vars);

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
  Variables<tmpl::list<
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2,
      GeneralizedHarmonic::Tags::GaugeH<3>,
      GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<3>,
      GeneralizedHarmonic::Tags::Gamma1Gamma2,
      GeneralizedHarmonic::Tags::PiTwoNormals,
      GeneralizedHarmonic::Tags::NormalDotOneIndexConstraint,
      GeneralizedHarmonic::Tags::Gamma1Plus1,
      GeneralizedHarmonic::Tags::PiOneNormal<3>,
      GeneralizedHarmonic::Tags::GaugeConstraint<3, Frame::Inertial>,
      GeneralizedHarmonic::Tags::PhiTwoNormals<3>,
      GeneralizedHarmonic::Tags::ShiftDotThreeIndexConstraint<3>,
      GeneralizedHarmonic::Tags::MeshVelocityDotThreeIndexConstraint<3>,
      GeneralizedHarmonic::Tags::PhiOneNormal<3>,
      GeneralizedHarmonic::Tags::PiSecondIndexUp<3>,
      GeneralizedHarmonic::Tags::ThreeIndexConstraint<3, Frame::Inertial>,
      GeneralizedHarmonic::Tags::PhiFirstIndexUp<3>,
      GeneralizedHarmonic::Tags::PhiThirdIndexUp<3>,
      GeneralizedHarmonic::Tags::SpacetimeChristoffelFirstKindThirdIndexUp<3>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<3, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::DetSpatialMetric<DataVector>,
      gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelFirstKind<3, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeChristoffelSecondKind<3, Frame::Inertial, DataVector>,
      gr::Tags::TraceSpacetimeChristoffelFirstKind<3, Frame::Inertial,
                                                   DataVector>,
      gr::Tags::SpacetimeNormalVector<3, Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeNormalOneForm<3, Frame::Inertial, DataVector>,
      gr::Tags::DerivativesOfSpacetimeMetric<3, Frame::Inertial, DataVector>>>
      buffer(mesh.number_of_grid_points());

  GeneralizedHarmonic::TimeDerivative<3>::apply(
      make_not_null(&dt_psi), make_not_null(&dt_pi), make_not_null(&dt_phi),
      make_not_null(
          &get<GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>(
              buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma2>(
              buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::GaugeH<3>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<3>>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::Gamma1Gamma2>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::PiTwoNormals>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::NormalDotOneIndexConstraint>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::Gamma1Plus1>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::PiOneNormal<3>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::GaugeConstraint<3, Frame::Inertial>>(
              buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::PhiTwoNormals<3>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::ShiftDotThreeIndexConstraint<3>>(
              buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::MeshVelocityDotThreeIndexConstraint<
              3>>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::PhiOneNormal<3>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::PiSecondIndexUp<3>>(buffer)),
      make_not_null(&get<GeneralizedHarmonic::Tags::ThreeIndexConstraint<
                        3, Frame::Inertial>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::PhiFirstIndexUp<3>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::PhiThirdIndexUp<3>>(buffer)),
      make_not_null(
          &get<GeneralizedHarmonic::Tags::
                   SpacetimeChristoffelFirstKindThirdIndexUp<3>>(buffer)),
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
              buffer)),
      make_not_null(
          &get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
              buffer)),
      make_not_null(&get<gr::Tags::DetSpatialMetric<DataVector>>(buffer)),
      make_not_null(&get<gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial,
                                                          DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::SpacetimeChristoffelFirstKind<3, Frame::Inertial,
                                                       DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::SpacetimeChristoffelSecondKind<3, Frame::Inertial,
                                                        DataVector>>(buffer)),
      make_not_null(&get<gr::Tags::TraceSpacetimeChristoffelFirstKind<
                        3, Frame::Inertial, DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::SpacetimeNormalVector<3, Frame::Inertial, DataVector>>(
              buffer)),
      make_not_null(&get<gr::Tags::SpacetimeNormalOneForm<3, Frame::Inertial,
                                                          DataVector>>(buffer)),
      make_not_null(
          &get<gr::Tags::DerivativesOfSpacetimeMetric<3, Frame::Inertial,
                                                      DataVector>>(buffer)),
      d_psi, d_pi, d_phi, psi, pi, phi, gamma0, gamma1, gamma2, gauge_function,
      d4_H, std::nullopt);

  // Make sure the RHS is zero.
  CHECK_ITERABLE_CUSTOM_APPROX(
      dt_psi, make_with_value<decltype(dt_psi)>(x, 0.0), numerical_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_pi, make_with_value<decltype(dt_pi)>(x, 0.0),
                               numerical_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      dt_phi, make_with_value<decltype(dt_phi)>(x, 0.0), numerical_approx);
}

namespace detail {
template <typename Tag, typename Frame>
using deriv = Tags::deriv<Tag, tmpl::size_t<3>, Frame>;

template <typename Tag, typename Solution, typename Frame>
auto time_derivative(const Solution& solution,
                     const tnsr::I<double, 3, Frame>& x, const double time,
                     const double dt) {
  typename Tag::type result{};
  for (auto it = result.begin(); it != result.end(); ++it) {
    const auto index = result.get_tensor_index(it);
    *it = numerical_derivative(
        [&index, &solution, &x](const std::array<double, 1>& t_array) {
          return std::array<double, 1>{
              {get<Tag>(solution.variables(x, t_array[0], tmpl::list<Tag>{}))
                   .get(index)}};
        },
        std::array<double, 1>{{time}}, 0, dt)[0];
  }
  return result;
}

template <typename Tag, typename Solution, typename Frame>
auto space_derivative(const Solution& solution,
                      const tnsr::I<double, 3, Frame>& x, const double time,
                      const double dx) {
  typename deriv<Tag, Frame>::type result{};
  for (auto it = result.begin(); it != result.end(); ++it) {
    const auto index = result.get_tensor_index(it);
    *it = numerical_derivative(
        [&index, &solution, &time, &x](const std::array<double, 1>& offset) {
          auto position = x;
          position.get(index[0]) += offset[0];
          return std::array<double, 1>{
              {get<Tag>(solution.variables(position, time, tmpl::list<Tag>{}))
                   .get(all_but_specified_element_of(index, 0))}};
        },
        std::array<double, 1>{{0.0}}, 0, dx)[0];
  }
  return result;
}
}  // namespace detail

/// Check the consistency of dependent quantities returned by a
/// solution.  This includes checking pointwise relations such as
/// consistency of the metric and inverse and comparing returned and
/// numerical derivatives.
template <typename Solution, typename Frame>
void verify_consistency(const Solution& solution, const double time,
                        const tnsr::I<double, 3, Frame>& position,
                        const double derivative_delta,
                        const double derivative_tolerance) {
  using Lapse = gr::Tags::Lapse<double>;
  using Shift = gr::Tags::Shift<3, Frame, double>;
  using SpatialMetric = gr::Tags::SpatialMetric<3, Frame, double>;
  using SqrtDetSpatialMetric = gr::Tags::SqrtDetSpatialMetric<double>;
  using InverseSpatialMetric = gr::Tags::InverseSpatialMetric<3, Frame, double>;
  using ExtrinsicCurvature = gr::Tags::ExtrinsicCurvature<3, Frame, double>;
  using tags =
      tmpl::list<SpatialMetric, SqrtDetSpatialMetric, InverseSpatialMetric,
                 ExtrinsicCurvature, Lapse, Shift, detail::deriv<Shift, Frame>,
                 Tags::dt<SpatialMetric>, detail::deriv<SpatialMetric, Frame>,
                 Tags::dt<Lapse>, Tags::dt<Shift>, detail::deriv<Lapse, Frame>>;

  auto derivative_approx = approx.epsilon(derivative_tolerance);

  const auto vars = solution.variables(position, time, tags{});

  const auto numerical_metric_det_and_inverse =
      determinant_and_inverse(get<SpatialMetric>(vars));
  CHECK_ITERABLE_APPROX(get(get<SqrtDetSpatialMetric>(vars)),
                        sqrt(get(numerical_metric_det_and_inverse.first)));
  CHECK_ITERABLE_APPROX(get<InverseSpatialMetric>(vars),
                        numerical_metric_det_and_inverse.second);

  CHECK_ITERABLE_APPROX(
      get<ExtrinsicCurvature>(vars),
      gr::extrinsic_curvature(get<Lapse>(vars), get<Shift>(vars),
                              get<detail::deriv<Shift, Frame>>(vars),
                              get<SpatialMetric>(vars),
                              get<Tags::dt<SpatialMetric>>(vars),
                              get<detail::deriv<SpatialMetric, Frame>>(vars)));

  CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::dt<Lapse>>(vars),
                               detail::time_derivative<Lapse>(
                                   solution, position, time, derivative_delta),
                               derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::dt<Shift>>(vars),
                               detail::time_derivative<Shift>(
                                   solution, position, time, derivative_delta),
                               derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::dt<SpatialMetric>>(vars),
                               detail::time_derivative<SpatialMetric>(
                                   solution, position, time, derivative_delta),
                               derivative_approx);

  CHECK_ITERABLE_CUSTOM_APPROX(
      SINGLE_ARG(get<detail::deriv<Lapse, Frame>>(vars)),
      detail::space_derivative<Lapse>(solution, position, time,
                                      derivative_delta),
      derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      SINGLE_ARG(get<detail::deriv<Shift, Frame>>(vars)),
      detail::space_derivative<Shift>(solution, position, time,
                                      derivative_delta),
      derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      SINGLE_ARG(get<detail::deriv<SpatialMetric, Frame>>(vars)),
      detail::space_derivative<SpatialMetric>(solution, position, time,
                                              derivative_delta),
      derivative_approx);
}
}  // namespace VerifyGrSolution
}  // namespace TestHelpers
