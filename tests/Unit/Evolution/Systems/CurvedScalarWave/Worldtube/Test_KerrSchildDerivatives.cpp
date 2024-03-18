// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/KerrSchildDerivatives.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

void test_derivative_inverse_metric() {
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(gen);
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  std::uniform_real_distribution<double> pos_dist{2., 10.};

  const gr::Solutions::KerrSchild kerr_schild(1., {{0., 0., 0.}},
                                              {{0., 0., 0.}});
  const auto test_point =
      make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&pos_dist), 1);
  const auto di_imetric = spatial_derivative_inverse_ks_metric(test_point);

  const std::array<double, 3> array_point{test_point.get(0), test_point.get(1),
                                          test_point.get(2)};
  const double dx = 1e-4;
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      for (size_t j = 0; j < 3; ++j) {
        auto inverse_metric_helper = [&kerr_schild, a,
                                      b](const std::array<double, 3>& point) {
          const tnsr::I<double, 3> tensor_point(point);
          const auto kerr_schild_quantities_local = kerr_schild.variables(
              tensor_point, 0.,
              tmpl::list<gr::Tags::Lapse<double>, gr::Tags::Shift<double, Dim>,
                         gr::Tags::InverseSpatialMetric<double, Dim>>{});
          const auto inverse_metric_local = gr::inverse_spacetime_metric(
              get<gr::Tags::Lapse<double>>(kerr_schild_quantities_local),
              get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities_local),
              get<gr::Tags::InverseSpatialMetric<double, Dim>>(
                  kerr_schild_quantities_local));
          return inverse_metric_local.get(a, b);
        };
        const auto numerical_deriv_imetric_j =
            numerical_derivative(inverse_metric_helper, array_point, j, dx);
        CHECK(di_imetric.get(j, a, b) ==
              local_approx(numerical_deriv_imetric_j));
      }
    }
  }
}


// All the derivatives are checked against finite difference calculations
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CurvedScalarWave.KerrSchildDerivatives",
    "[Unit][Evolution]") {
  test_derivative_inverse_metric();
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
