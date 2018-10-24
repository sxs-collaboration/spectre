// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp" // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
void test_three_index_constraint(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::iaa<DataType, SpatialDim, Frame> (*)(
          const tnsr::iaa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &GeneralizedHarmonic::three_index_constraint<SpatialDim, Frame,
                                                       DataType>),
      "numpy", "subtract", {{{-10.0, 10.0}}}, used_for_size);
}

// Test the return-by-value gauge constraint function using random values
template <size_t SpatialDim, typename Frame, typename DataType>
void test_gauge_constraint_random(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::a<DataType, SpatialDim, Frame> (*)(
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::a<DataType, SpatialDim, Frame>&,
          const tnsr::A<DataType, SpatialDim, Frame>&,
          const tnsr::II<DataType, SpatialDim, Frame>&,
          const tnsr::AA<DataType, SpatialDim, Frame>&,
          const tnsr::aa<DataType, SpatialDim, Frame>&,
          const tnsr::iaa<DataType, SpatialDim, Frame>&)>(
          &GeneralizedHarmonic::gauge_constraint<SpatialDim, Frame, DataType>),
      "TestFunctions", "gauge_constraint", {{{-10.0, 10.0}}}, used_for_size);
}

// Test test return-by-reference gauge constraint by comparing to Kerr-Schild
template <typename Solution>
void test_gauge_constraint_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Check vs. time-independent analytic solution
  // Set up grid
  Mesh<3> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};

  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

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
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<3>>>(vars);
  const auto& d_spatial_metric =
      get<typename Solution::template DerivSpatialMetric<DataVector>>(vars);

  // Get ingredients for computing the gauge constraint
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto& inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);
  const auto& gauge_function = GeneralizedHarmonic::gauge_source(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      trace(gr::extrinsic_curvature(lapse, shift, d_shift, spatial_metric,
                                    dt_spatial_metric, d_spatial_metric),
            inverse_spatial_metric),
      trace_last_indices(gr::christoffel_first_kind(d_spatial_metric),
                         inverse_spatial_metric));
  const auto& phi = GeneralizedHarmonic::phi(lapse, d_lapse, shift, d_shift,
                                             spatial_metric, d_spatial_metric);
  const auto& pi = GeneralizedHarmonic::pi(
      lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric, phi);
  const auto& normal_one_form =
      gr::spacetime_normal_one_form<3, Frame::Inertial>(lapse);
  const auto& normal_vector = gr::spacetime_normal_vector(lapse, shift);

  // Get the constraint, and check that it vanishes
  auto constraint =
      make_with_value<tnsr::a<DataVector, 3, Frame::Inertial>>(x, 0.0);
  const gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
      constraint_pointer = &(constraint);
  GeneralizedHarmonic::gauge_constraint(
      constraint_pointer, gauge_function, normal_one_form, normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  CHECK_ITERABLE_APPROX(constraint,
                        make_with_value<decltype(constraint)>(x, 0.0));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ThreeIndexConstraint",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  test_three_index_constraint<1, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<1, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<1, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<1, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_three_index_constraint<2, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<2, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<2, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<2, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_three_index_constraint<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_three_index_constraint<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_three_index_constraint<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.GaugeConstraint",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  // Test the gauge constraint against Kerr Schild
  const double mass = 1.4;
  const std::array<double, 3> spin{{0.4, 0.3, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  test_gauge_constraint_analytic(solution, grid_size, lower_bound, upper_bound);

  // Test the gauge constraint with random numbers
  test_gauge_constraint_random<1, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<1, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<1, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<1, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_gauge_constraint_random<2, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<2, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<2, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<2, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());

  test_gauge_constraint_random<3, Frame::Grid, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<3, Frame::Inertial, DataVector>(
      DataVector(4, std::numeric_limits<double>::signaling_NaN()));
  test_gauge_constraint_random<3, Frame::Grid, double>(
      std::numeric_limits<double>::signaling_NaN());
  test_gauge_constraint_random<3, Frame::Inertial, double>(
      std::numeric_limits<double>::signaling_NaN());
}
