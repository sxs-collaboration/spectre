// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/AnalyticSolutions/AnalyticDataHelpers.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.RobinsonTrautman",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> radius_dist{10.0, 20.0};
  UniformCustomDistribution<double> parameter_dist{0.001, 0.01};
  const double extraction_radius = radius_dist(gen);
  const double start_time = 0.0;
  // the system is somewhat stiff, so to save time in the test, we only evolve
  // for a short time compared to the extraction radius.
  const double time = 0.02;
  const size_t l_max = 16;
  // the l=2 modes are what we should care most about, so we test with only the
  // (2, -2) entry populated.
  std::vector<std::complex<double>> modes{0.0, 0.0, 0.0, 0.0,
                                          parameter_dist(gen)};
  Options::Context context{};
  const RobinsonTrautman boundary_solution{
      std::move(modes), extraction_radius, l_max, 1.0e-13, start_time, context};

  const auto boundary_data = boundary_solution.variables(
      l_max, time,
      tmpl::list<
          Tags::Dr<Tags::CauchyCartesianCoords>,
          gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>,
          ::Tags::dt<
              gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>,
          GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>, Tags::News>{});

  const auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
          boundary_data);


  CartesianiSphericalJ inverse_jacobian{get<0, 0>(spacetime_metric).size()};
  boundary_solution.inverse_jacobian(make_not_null(&inverse_jacobian), l_max);

  const auto bondi_quantities =
      TestHelpers::extract_bondi_scalars_from_cartesian_metric(
          spacetime_metric, inverse_jacobian, extraction_radius);

  const auto& dt_spacetime_metric = get<
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>(
      boundary_data);
  const auto dt_bondi_quantities =
      TestHelpers::extract_dt_bondi_scalars_from_cartesian_metric(
          dt_spacetime_metric, spacetime_metric, inverse_jacobian,
          extraction_radius);
  CartesianiSphericalJ dr_inverse_jacobian{get<0, 0>(spacetime_metric).size()};
  boundary_solution.dr_inverse_jacobian(make_not_null(&dr_inverse_jacobian),
                                        l_max);
  const auto& d_spacetime_metric =
      get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(boundary_data);
  const auto& dr_cartesian_coordinates =
      get<Tags::Dr<Tags::CauchyCartesianCoords>>(boundary_data);
  tnsr::aa<DataVector, 3> dr_spacetime_metric{
      get<0, 0>(spacetime_metric).size(), 0.0};
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      for (size_t i = 0; i < 3; ++i) {
        dr_spacetime_metric.get(a, b) +=
            dr_cartesian_coordinates.get(i) * d_spacetime_metric.get(i, a, b);
      }
    }
  }
  const auto dr_bondi_quantities =
      TestHelpers::extract_dr_bondi_scalars_from_cartesian_metric(
          dr_spacetime_metric, spacetime_metric, inverse_jacobian,
          dr_inverse_jacobian, extraction_radius);

  Approx bondi_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  // we use the specific form of the Robinson-Trautman solution to determine
  // the generating scalar, then use that to check that the remaining parts
  // of the metric are appropriately related.
  const SpinWeighted<ComplexDataVector, 0> inferred_rt_scalar =
      exp(-2.0 * get(get<Tags::BondiBeta>(bondi_quantities)));
  const SpinWeighted<ComplexDataVector, 1> expected_bondi_u =
      Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::Eth>(
          l_max, 1, inferred_rt_scalar) /
      extraction_radius;
  const SpinWeighted<ComplexDataVector, 0> expected_bondi_w =
      (inferred_rt_scalar +
       Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::EthEthbar>(
           l_max, 1, inferred_rt_scalar) -
       1.0) /
          extraction_radius -
      2.0 / square(extraction_radius * inferred_rt_scalar);
  const SpinWeighted<ComplexDataVector, 2> expected_bondi_j{
      inferred_rt_scalar.size(), 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(get(get<Tags::BondiU>(bondi_quantities)),
                               expected_bondi_u, bondi_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(get<Tags::BondiW>(bondi_quantities)),
                               expected_bondi_w, bondi_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(get<Tags::BondiJ>(bondi_quantities)),
                               expected_bondi_j, bondi_approx);

  const auto ethbar_ethbar_rt_scalar =
      Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::EthbarEthbar>(
          l_max, 1, inferred_rt_scalar);
  SpinWeighted<ComplexDataVector, 0> expected_dt_rt_scalar =
      (-pow<4>(inferred_rt_scalar) *
           Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::EthEth>(
               l_max, 1, ethbar_ethbar_rt_scalar) +
       pow<3>(inferred_rt_scalar) * ethbar_ethbar_rt_scalar *
           conj(ethbar_ethbar_rt_scalar)) /
      12.0;
  Spectral::Swsh::filter_swsh_boundary_quantity(
      make_not_null(&expected_dt_rt_scalar), l_max, l_max - 3);

  // the evolution equation involves a fourth angular derivative, so requires a
  // somewhat looser tolerance
  Approx derivative_bondi_approx =
      Approx::custom()
      .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
      .scale(1.0);

  const SpinWeighted<ComplexDataVector, 0> expected_dt_bondi_beta =
      expected_dt_rt_scalar / (2.0 * inferred_rt_scalar);
  const SpinWeighted<ComplexDataVector, 1> expected_dt_bondi_u =
      Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::Eth>(
          l_max, 1, expected_dt_rt_scalar) /
      extraction_radius;
  const SpinWeighted<ComplexDataVector, 0> expected_dt_bondi_w =
      (expected_dt_rt_scalar +
       Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::EthEthbar>(
           l_max, 1, expected_dt_rt_scalar)) /
          extraction_radius +
      4.0 * expected_dt_rt_scalar /
          (square(extraction_radius) * pow<3>(inferred_rt_scalar));
  const SpinWeighted<ComplexDataVector, 2> expected_dt_bondi_j{
      inferred_rt_scalar.size(), 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<::Tags::dt<Tags::BondiBeta>>(dt_bondi_quantities)),
      expected_dt_bondi_beta, derivative_bondi_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<::Tags::dt<Tags::BondiU>>(dt_bondi_quantities)),
      expected_dt_bondi_u, derivative_bondi_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<::Tags::dt<Tags::BondiW>>(dt_bondi_quantities)),
      expected_dt_bondi_w, derivative_bondi_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<::Tags::dt<Tags::BondiJ>>(dt_bondi_quantities)),
      expected_dt_bondi_j, derivative_bondi_approx);

  const SpinWeighted<ComplexDataVector, 0> expected_dr_bondi_beta =
      -expected_dt_bondi_beta;
  const SpinWeighted<ComplexDataVector, 1> expected_dr_bondi_u =
      -expected_dt_bondi_u -
      Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::Eth>(
          l_max, 1, inferred_rt_scalar) /
          square(extraction_radius);
  const SpinWeighted<ComplexDataVector, 0> expected_dr_bondi_w =
      -expected_dt_bondi_w -
      (inferred_rt_scalar +
       Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::EthEthbar>(
           l_max, 1, inferred_rt_scalar) -
       1.0) /
          square(extraction_radius) +
      4.0 / (pow<3>(extraction_radius) * square(inferred_rt_scalar));
  const SpinWeighted<ComplexDataVector, 2> expected_dr_bondi_j{
      inferred_rt_scalar.size(), 0.0};

  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<Tags::Dr<Tags::BondiBeta>>(dr_bondi_quantities)),
      expected_dr_bondi_beta, bondi_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<Tags::Dr<Tags::BondiU>>(dr_bondi_quantities)),
      expected_dr_bondi_u, bondi_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<Tags::Dr<Tags::BondiW>>(dr_bondi_quantities)),
      expected_dr_bondi_w, bondi_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(get<Tags::Dr<Tags::BondiJ>>(dr_bondi_quantities)),
      expected_dr_bondi_j, bondi_approx);

  const auto& news = get(get<Tags::News>(boundary_data));
  const SpinWeighted<ComplexDataVector, -2> expected_news =
      Spectral::Swsh::angular_derivative<Spectral::Swsh::Tags::EthbarEthbar>(
          l_max, 1, inferred_rt_scalar) /
      inferred_rt_scalar;
  CHECK_ITERABLE_CUSTOM_APPROX(news, expected_news, bondi_approx);
}
}  // namespace Cce::Solutions
