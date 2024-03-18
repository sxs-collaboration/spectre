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

void test_derivative_metric() {
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(gen);
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  std::uniform_real_distribution<double> pos_dist{2., 10.};

  const gr::Solutions::KerrSchild kerr_schild(1., {{0., 0., 0.}},
                                              {{0., 0., 0.}});
  const auto test_point =
      make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&pos_dist), 1);
  const auto kerr_schild_quantities = kerr_schild.variables(
      test_point, 0.,
      tmpl::list<gr::Tags::Lapse<double>, gr::Tags::Shift<double, Dim>,
                 gr::Tags::SpatialMetric<double, Dim>,
                 gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>{});
  const auto metric = gr::spacetime_metric(
      get<gr::Tags::Lapse<double>>(kerr_schild_quantities),
      get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities),
      get<gr::Tags::SpatialMetric<double, Dim>>(kerr_schild_quantities));

  const auto& di_imetric = spatial_derivative_inverse_ks_metric(test_point);
  const auto di_metric = spatial_derivative_ks_metric(metric, di_imetric);

  const std::array<double, 3> array_point{test_point.get(0), test_point.get(1),
                                          test_point.get(2)};
  const double dx = 1e-4;

  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      for (size_t j = 0; j < 3; ++j) {
        const auto metric_helper = [&kerr_schild, a,
                                    b](const std::array<double, 3>& point) {
          const tnsr::I<double, 3> tensor_point(point);
          const auto kerr_schild_quantities_local = kerr_schild.variables(
              tensor_point, 0.,
              tmpl::list<gr::Tags::Lapse<double>, gr::Tags::Shift<double, Dim>,
                         gr::Tags::SpatialMetric<double, Dim>>{});
          const auto metric_local = gr::spacetime_metric(
              get<gr::Tags::Lapse<double>>(kerr_schild_quantities_local),
              get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities_local),
              get<gr::Tags::SpatialMetric<double, Dim>>(
                  kerr_schild_quantities_local));
          return metric_local.get(a, b);
        };
        const auto numerical_deriv_metric_j =
            numerical_derivative(metric_helper, array_point, j, dx);
        CHECK(di_metric.get(j, a, b) == local_approx(numerical_deriv_metric_j));
      }
    }
  }
}

void test_second_derivative_inverse_metric() {
  MAKE_GENERATOR(gen);
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  std::uniform_real_distribution<double> pos_dist{2., 10.};

  const gr::Solutions::KerrSchild kerr_schild(1., {{0., 0., 0.}},
                                              {{0., 0., 0.}});

  const auto test_point =
      make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&pos_dist), 1);

  const auto& dij_imetric =
      second_spatial_derivative_inverse_ks_metric(test_point);

  const std::array<double, 3> array_point{test_point.get(0), test_point.get(1),
                                          test_point.get(2)};
  const double dx = 1e-4;
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k <= j; ++k) {
          const auto di_imetric_helper =
              [a, b, j](const std::array<double, 3>& point) {
                const tnsr::I<double, 3> tensor_point(point);
                const auto di_imetric_local =
                    spatial_derivative_inverse_ks_metric(tensor_point);
                return di_imetric_local.get(j, a, b);
              };
          const auto second_numerical_deriv_imetric_k =
              numerical_derivative(di_imetric_helper, array_point, k, dx);
          CHECK(dij_imetric.get(k, j, a, b) ==
                local_approx(second_numerical_deriv_imetric_k));
        }
      }
    }
  }
}

void test_second_derivative_metric() {
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(gen);
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  std::uniform_real_distribution<double> pos_dist{2., 10.};
  const gr::Solutions::KerrSchild kerr_schild(1., {{0., 0., 0.}},
                                              {{0., 0., 0.}});

  const auto test_point =
      make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&pos_dist), 1);
  const auto kerr_schild_quantities = kerr_schild.variables(
      test_point, 0.,
      tmpl::list<gr::Tags::Lapse<double>, gr::Tags::Shift<double, Dim>,
                 gr::Tags::SpatialMetric<double, Dim>,
                 gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>{});
  const auto metric = gr::spacetime_metric(
      get<gr::Tags::Lapse<double>>(kerr_schild_quantities),
      get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities),
      get<gr::Tags::SpatialMetric<double, Dim>>(kerr_schild_quantities));

  const auto& di_imetric = spatial_derivative_inverse_ks_metric(test_point);
  const auto& dij_imetric =
      second_spatial_derivative_inverse_ks_metric(test_point);
  const auto di_metric = spatial_derivative_ks_metric(metric, di_imetric);
  const auto& dij_metric = second_spatial_derivative_metric(
      metric, di_metric, di_imetric, dij_imetric);

  const std::array<double, 3> array_point{test_point.get(0), test_point.get(1),
                                          test_point.get(2)};
  const double dx = 1e-4;
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k <= j; ++k) {
          const auto di_metric_helper = [&kerr_schild, j, a,
                                         b](const std::array<double, 3>&
                                                point) {
            const tnsr::I<double, 3> tensor_point(point);
            const auto kerr_schild_quantities_local = kerr_schild.variables(
                tensor_point, 0.,
                tmpl::list<
                    gr::Tags::Lapse<double>, gr::Tags::Shift<double, Dim>,
                    gr::Tags::SpatialMetric<double, Dim>,
                    gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>{});
            const auto metric_local = gr::spacetime_metric(
                get<gr::Tags::Lapse<double>>(kerr_schild_quantities_local),
                get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities_local),
                get<gr::Tags::SpatialMetric<double, Dim>>(
                    kerr_schild_quantities_local));
            const auto di_imetric_local =
                spatial_derivative_inverse_ks_metric(tensor_point);
            const auto di_metric_local =
                spatial_derivative_ks_metric(metric_local, di_imetric_local);
            return di_metric_local.get(j, a, b);
          };

          const auto second_numerical_deriv_metric_k =
              numerical_derivative(di_metric_helper, array_point, k, dx);
          CHECK(dij_metric.get(k, j, a, b) ==
                local_approx(second_numerical_deriv_metric_k));
        }
      }
    }
  }
}

void test_derivative_christoffel() {
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(gen);
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  std::uniform_real_distribution<double> pos_dist{2., 10.};

  const gr::Solutions::KerrSchild kerr_schild(1., {{0., 0., 0.}},
                                              {{0., 0., 0.}});

  const auto test_point =
      make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&pos_dist), 1);
  const auto kerr_schild_quantities = kerr_schild.variables(
      test_point, 0.,
      tmpl::list<gr::Tags::Lapse<double>, gr::Tags::Shift<double, Dim>,
                 gr::Tags::SpatialMetric<double, Dim>,
                 gr::Tags::InverseSpatialMetric<double, Dim>,
                 gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>{});
  const auto metric = gr::spacetime_metric(
      get<gr::Tags::Lapse<double>>(kerr_schild_quantities),
      get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities),
      get<gr::Tags::SpatialMetric<double, Dim>>(kerr_schild_quantities));
  const auto inverse_metric = gr::inverse_spacetime_metric(
      get<gr::Tags::Lapse<double>>(kerr_schild_quantities),
      get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities),
      get<gr::Tags::InverseSpatialMetric<double, Dim>>(kerr_schild_quantities));

  const auto& di_imetric = spatial_derivative_inverse_ks_metric(test_point);
  const auto& dij_imetric =
      second_spatial_derivative_inverse_ks_metric(test_point);
  const auto di_metric = spatial_derivative_ks_metric(metric, di_imetric);
  const auto& dij_metric = second_spatial_derivative_metric(
      metric, di_metric, di_imetric, dij_imetric);
  const auto di_christoffel = spatial_derivative_christoffel(
      di_metric, dij_metric, inverse_metric, di_imetric);

  const std::array<double, 3> array_point{test_point.get(0), test_point.get(1),
                                          test_point.get(2)};
  const double dx = 1e-4;
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < 4; ++b) {
      for (size_t c = 0; c < 4; ++c) {
        for (size_t j = 0; j < 3; ++j) {
          const auto christoffel_helper =
              [&kerr_schild, a, b, c](const std::array<double, 3>& point) {
                const tnsr::I<double, 3> tensor_point(point);
                const auto kerr_schild_quantities_local = kerr_schild.variables(
                    tensor_point, 0.,
                    tmpl::list<gr::Tags::SpacetimeChristoffelSecondKind<
                        double, Dim>>{});
                const auto& christoffel_local =
                    get<gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>(
                        kerr_schild_quantities_local);
                return christoffel_local.get(a, b, c);
              };

          const auto numerical_deriv_christoffel_j =
              numerical_derivative(christoffel_helper, array_point, j, dx);
          CHECK(di_christoffel.get(j, a, b, c) ==
                local_approx(numerical_deriv_christoffel_j));
        }
      }
    }
  }
}

void test_derivative_contracted_christoffel() {
  static constexpr size_t Dim = 3;
  MAKE_GENERATOR(gen);
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  std::uniform_real_distribution<double> pos_dist{2., 10.};

  const gr::Solutions::KerrSchild kerr_schild(1., {{0., 0., 0.}},
                                              {{0., 0., 0.}});

  const auto test_point =
      make_with_random_values<tnsr::I<double, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&pos_dist), 1);
  const auto di_contracted_christoffel =
      spatial_derivative_ks_contracted_christoffel(test_point);

  const std::array<double, 3> array_point{test_point.get(0), test_point.get(1),
                                          test_point.get(2)};
  const double dx = 1e-4;
  for (size_t a = 0; a < 4; ++a) {
    for (size_t j = 0; j < 3; ++j) {
      const auto contracted_christoffel_helper =
          [&kerr_schild, a](const std::array<double, 3>& point) {
            const tnsr::I<double, 3> tensor_point(point);
            const auto kerr_schild_quantities_local = kerr_schild.variables(
                tensor_point, 0.,
                tmpl::list<
                    gr::Tags::Lapse<double>, gr::Tags::Shift<double, Dim>,
                    gr::Tags::SpatialMetric<double, Dim>,
                    gr::Tags::InverseSpatialMetric<double, Dim>,
                    gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>{});
            const auto imetric_local = gr::inverse_spacetime_metric(
                get<gr::Tags::Lapse<double>>(kerr_schild_quantities_local),
                get<gr::Tags::Shift<double, Dim>>(kerr_schild_quantities_local),
                get<gr::Tags::InverseSpatialMetric<double, Dim>>(
                    kerr_schild_quantities_local));
            const auto& christoffel_local =
                get<gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>(
                    kerr_schild_quantities_local);
            const auto contracted_christoffel_local =
                trace_last_indices(christoffel_local, imetric_local);
            return contracted_christoffel_local.get(a);
          };
      const auto numerical_deriv_contracted_christoffel_j =
          numerical_derivative(contracted_christoffel_helper, array_point, j,
                               dx);
      CHECK(di_contracted_christoffel.get(j, a) ==
            local_approx(numerical_deriv_contracted_christoffel_j));
    }
  }
}

// All the derivatives are checked against finite difference calculations
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CurvedScalarWave.KerrSchildDerivatives",
    "[Unit][Evolution]") {
  test_derivative_inverse_metric();
  test_derivative_metric();
  test_second_derivative_inverse_metric();
  test_second_derivative_metric();
  test_derivative_christoffel();
  test_derivative_contracted_christoffel();
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
