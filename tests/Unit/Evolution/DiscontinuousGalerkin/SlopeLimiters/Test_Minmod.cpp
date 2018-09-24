// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"
#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare SlopeLimiters::Minmod
// IWYU pragma: no_forward_declare Tensor

namespace {
struct scalar : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Scalar"; }
};

template <size_t VolumeDim>
struct vector : db::SimpleTag {
  using type = tnsr::I<DataVector, VolumeDim>;
  static std::string name() noexcept { return "Vector"; }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.Minmod.Options",
                  "[SlopeLimiters][Unit]") {
  const auto lambda_pi1_default =
      test_creation<SlopeLimiters::Minmod<1, tmpl::list<scalar>>>(
          "  Type: LambdaPi1");
  const auto lambda_pi1_m0 =
      test_creation<SlopeLimiters::Minmod<1, tmpl::list<scalar>>>(
          "  Type: LambdaPi1\n  TvbmConstant: 0.0");
  const auto lambda_pi1_m1 =
      test_creation<SlopeLimiters::Minmod<1, tmpl::list<scalar>>>(
          "  Type: LambdaPi1\n  TvbmConstant: 1.0");
  const auto muscl_default =
      test_creation<SlopeLimiters::Minmod<1, tmpl::list<scalar>>>(
          "  Type: Muscl");

  // Test default TVBM value, operator==, and operator!=
  CHECK(lambda_pi1_default == lambda_pi1_m0);
  CHECK(lambda_pi1_default != lambda_pi1_m1);
  CHECK(lambda_pi1_default != muscl_default);

  test_creation<SlopeLimiters::Minmod<1, tmpl::list<scalar>>>(
      "  Type: LambdaPiN");
  test_creation<SlopeLimiters::Minmod<2, tmpl::list<scalar>>>(
      "  Type: LambdaPiN");
  test_creation<SlopeLimiters::Minmod<3, tmpl::list<scalar, vector<3>>>>(
      "  Type: LambdaPiN");
}

// [[OutputRegex, Failed to convert "BadType" to MinmodType]]
SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.Minmod.OptionParseError",
                  "[SlopeLimiters][Unit]") {
  ERROR_TEST();
  test_creation<SlopeLimiters::Minmod<1, tmpl::list<scalar>>>(
      "  Type: BadType");
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.Minmod.Serialization",
                  "[SlopeLimiters][Unit]") {
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod(
      SlopeLimiters::MinmodType::LambdaPi1);
  test_serialization(minmod);
}

namespace {
template <size_t VolumeDim>
Neighbors<VolumeDim> make_neighbor_with_id(const size_t id) noexcept {
  return {std::unordered_set<ElementId<VolumeDim>>{ElementId<VolumeDim>(id)},
          OrientationMap<VolumeDim>{}};
}

// Construct an element with one neighbor in each direction.
template <size_t VolumeDim>
Element<VolumeDim> make_element() noexcept;

template <>
Element<1> make_element() noexcept {
  return Element<1>{
      ElementId<1>{0},
      Element<1>::Neighbors_t{
          {Direction<1>::lower_xi(), make_neighbor_with_id<1>(1)},
          {Direction<1>::upper_xi(), make_neighbor_with_id<1>(2)}}};
}

template <>
Element<2> make_element() noexcept {
  return Element<2>{
      ElementId<2>{0},
      Element<2>::Neighbors_t{
          {Direction<2>::lower_xi(), make_neighbor_with_id<2>(1)},
          {Direction<2>::upper_xi(), make_neighbor_with_id<2>(2)},
          {Direction<2>::lower_eta(), make_neighbor_with_id<2>(3)},
          {Direction<2>::upper_eta(), make_neighbor_with_id<2>(4)}}};
}

template <>
Element<3> make_element() noexcept {
  return Element<3>{
      ElementId<3>{0},
      Element<3>::Neighbors_t{
          {Direction<3>::lower_xi(), make_neighbor_with_id<3>(1)},
          {Direction<3>::upper_xi(), make_neighbor_with_id<3>(2)},
          {Direction<3>::lower_eta(), make_neighbor_with_id<3>(3)},
          {Direction<3>::upper_eta(), make_neighbor_with_id<3>(4)},
          {Direction<3>::lower_zeta(), make_neighbor_with_id<3>(5)},
          {Direction<3>::upper_zeta(), make_neighbor_with_id<3>(6)}}};
}

std::unordered_map<Direction<1>, Scalar<double>> make_neighbor_means(
    const double left, const double right) noexcept {
  return std::unordered_map<Direction<1>, Scalar<double>>{
      {Direction<1>::lower_xi(), Scalar<double>(left)},
      {Direction<1>::upper_xi(), Scalar<double>(right)}};
}

// Set neighbor sizes to local size, corresponding to uniform elements.
template <size_t VolumeDim>
std::unordered_map<Direction<VolumeDim>, tnsr::I<double, VolumeDim>>
make_neighbor_sizes_from_local_size(
    const tnsr::I<double, VolumeDim>& local_size) noexcept;

template <>
std::unordered_map<Direction<1>, tnsr::I<double, 1>>
make_neighbor_sizes_from_local_size(
    const tnsr::I<double, 1>& local_size) noexcept {
  return std::unordered_map<Direction<1>, tnsr::I<double, 1>>{
      {Direction<1>::lower_xi(), local_size},
      {Direction<1>::upper_xi(), local_size}};
}

template <>
std::unordered_map<Direction<2>, tnsr::I<double, 2>>
make_neighbor_sizes_from_local_size(
    const tnsr::I<double, 2>& local_size) noexcept {
  return std::unordered_map<Direction<2>, tnsr::I<double, 2>>{
      {Direction<2>::lower_xi(), local_size},
      {Direction<2>::upper_xi(), local_size},
      {Direction<2>::lower_eta(), local_size},
      {Direction<2>::upper_eta(), local_size}};
}

template <>
std::unordered_map<Direction<3>, tnsr::I<double, 3>>
make_neighbor_sizes_from_local_size(
    const tnsr::I<double, 3>& local_size) noexcept {
  return std::unordered_map<Direction<3>, tnsr::I<double, 3>>{
      {Direction<3>::lower_xi(), local_size},
      {Direction<3>::upper_xi(), local_size},
      {Direction<3>::lower_eta(), local_size},
      {Direction<3>::upper_eta(), local_size},
      {Direction<3>::lower_zeta(), local_size},
      {Direction<3>::upper_zeta(), local_size}};
}

void test_limiter_activates_work(
    const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
    const scalar::type& input,
    const std::unordered_map<Direction<1>, Scalar<double>>& neighbor_means,
    const Element<1>& element, const Mesh<1>& mesh,
    const tnsr::I<DataVector, 1, Frame::Logical>& logical_coords,
    const tnsr::I<double, 1>& element_size,
    const std::unordered_map<Direction<1>, tnsr::I<double, 1>>& neighbor_sizes,
    const double expected_slope) noexcept {
  auto input_to_limit = input;
  const bool limiter_activated =
      minmod.apply(make_not_null(&input_to_limit), neighbor_means, element,
                   mesh, logical_coords, element_size, neighbor_sizes);
  CHECK(limiter_activated);
  const scalar::type expected_output = [&logical_coords, &mesh ](
      const scalar::type& in, const double slope) noexcept {
    const double mean = mean_value(get(in), mesh);
    return scalar::type(mean + get<0>(logical_coords) * slope);
  }
  (input, expected_slope);
  CHECK_ITERABLE_APPROX(input_to_limit, expected_output);
}

void test_limiter_does_not_activate_work(
    const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
    const scalar::type& input,
    const std::unordered_map<Direction<1>, Scalar<double>>& neighbor_means,
    const Element<1>& element, const Mesh<1>& mesh,
    const tnsr::I<DataVector, 1, Frame::Logical>& logical_coords,
    const tnsr::I<double, 1>& element_size,
    const std::unordered_map<Direction<1>, tnsr::I<double, 1>>&
        neighbor_sizes) noexcept {
  auto input_to_limit = input;
  const bool limiter_activated =
      minmod.apply(make_not_null(&input_to_limit), neighbor_means, element,
                   mesh, logical_coords, element_size, neighbor_sizes);
  // The limiter can report an activation even if the solution was not modified.
  // This occurs when a pure-linear solution is represented on a higher-than-
  // linear order grid, so that the limiter's linearizing step doesn't actually
  // change the solution. If the slope does not need to be reduced, then the
  // limiter has no effect, but the limiter itself doesn't know this, because it
  // doesn't know the original solution had no higher-order content. Therefore,
  // the limiter reports an activation even though the solution was not changed.
  //
  // This effect also depends on the limiter type: for linear solutions that do
  // not need to have their sloped reduced, the LambdaPiN limiter's troubled-
  // cell detection will kick in and skip limiting altogether. This means that
  // when using LambdaPiN, we should not see the spurious activations described
  // in the previous paragraph.
  //
  // Note that it is somewhat artificial to have a pure-linear solution on a
  // higher-than-linear order element; this scenario should only occur in code
  // testing or certain test problems with very clean initial data.
  if (mesh.extents(0) > 2 and
      minmod.minmod_type() != SlopeLimiters::MinmodType::LambdaPiN) {
    CHECK(limiter_activated);
  } else {
    CHECK_FALSE(limiter_activated);
  }
  CHECK_ITERABLE_APPROX(input_to_limit, input);
}

void test_limiter_action_on_constant_function(
    const size_t number_of_grid_points,
    const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod) noexcept {
  INFO("Testing constant function...");
  CAPTURE(number_of_grid_points);
  const auto element = make_element<1>();
  const auto mesh = Mesh<1>(number_of_grid_points, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const double value = 0.3;
  const auto input =
      make_with_value<scalar::type>(get<0>(logical_coords), value);
  const auto element_size = tnsr::I<double, 1>{1.2};
  const auto neighbor_sizes = make_neighbor_sizes_from_local_size(element_size);
  for (const double left : {-0.4, value, 1.2}) {
    for (const double right : {0.2, value, 0.9}) {
      test_limiter_does_not_activate_work(
          minmod, input, make_neighbor_means(left, right), element, mesh,
          logical_coords, element_size, neighbor_sizes);
    }
  }
}

void test_limiter_action_on_linear_function(
    const size_t number_of_grid_points,
    const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod) noexcept {
  INFO("Testing linear function...");
  CAPTURE(number_of_grid_points);
  const auto element = make_element<1>();
  const auto mesh = Mesh<1>(number_of_grid_points, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = tnsr::I<double, 1>{2.0};
  const auto neighbor_sizes = make_neighbor_sizes_from_local_size(element_size);

  const auto test_limiter_activates =
      [
        &minmod, &element, &mesh, &logical_coords, &element_size,
        &neighbor_sizes
      ](const scalar::type& local_input, const double left, const double right,
        const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes, expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [
        &minmod, &element, &mesh, &logical_coords, &element_size,
        &neighbor_sizes
      ](const scalar::type& local_input, const double left,
        const double right) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes);
  };

  // With a MUSCL limiter, the largest allowed slope is half as big as for a
  // LambdaPi1 or LambdaPiN limiter. We can re-use the same test cases by
  // correspondingly scaling the slope:
  const double muscl_slope_factor =
      (minmod.minmod_type() == SlopeLimiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  // Test a positive-slope function
  {
    const auto func = [&muscl_slope_factor](
        const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
      return 3.6 + 1.2 * muscl_slope_factor * get<0>(coords);
    };
    const auto input = scalar::type(func(logical_coords));

    // Steepness test
    // Limiter does not reduce slope if (difference of means) > (local slope),
    // but reduces slope if (difference of means) < (local slope)
    test_limiter_does_not_activate(input, 2.0, 6.0);
    test_limiter_does_not_activate(input, 2.4 - eps, 4.8 + eps);
    test_limiter_activates(input, 2.6, 6.0, 1.0 * muscl_slope_factor);
    test_limiter_activates(input, 2.0, 4.0, 0.4 * muscl_slope_factor);
    test_limiter_activates(input, 2.6, 4.0, 0.4 * muscl_slope_factor);

    // Local extremum test
    // Limiter flattens slope if both neighbors are above (below) the mean
    test_limiter_activates(input, 1.0, 2.0, 0.0);
    test_limiter_activates(input, 6.0, 9.0, 0.0);

    // Oscillation test
    // Limiter flattens slope if sign(difference of means) != sign(local slope)
    test_limiter_activates(input, 3.9, 2.7, 0.0);
  }

  // Test a negative-slope function
  {
    const auto func = [&muscl_slope_factor](
        const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
      return -0.4 - 0.8 * muscl_slope_factor * get<0>(coords);
    };
    const auto input = scalar::type(func(logical_coords));

    // Steepness test
    test_limiter_does_not_activate(input, 0.9, -2.3);
    test_limiter_does_not_activate(input, 0.4 + eps, -1.2 - eps);
    test_limiter_activates(input, 0.2, -1.2, -0.6 * muscl_slope_factor);
    test_limiter_activates(input, 0.4, -0.5, -0.1 * muscl_slope_factor);
    test_limiter_activates(input, 0.2, -0.5, -0.1 * muscl_slope_factor);

    // Local extremum test
    test_limiter_activates(input, 1.3, -0.1, 0.0);
    test_limiter_activates(input, -3.2, -0.8, 0.0);

    // Oscillation test
    test_limiter_activates(input, -2.3, 0.2, 0.0);
  }
}

void test_limiter_action_on_quadratic_function(
    const size_t number_of_grid_points,
    const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod) noexcept {
  INFO("Testing quadratic function...");
  CAPTURE(number_of_grid_points);
  const auto element = make_element<1>();
  const auto mesh = Mesh<1>(number_of_grid_points, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = tnsr::I<double, 1>{2.0};
  const auto neighbor_sizes = make_neighbor_sizes_from_local_size(element_size);

  const auto test_limiter_activates =
      [
        &minmod, &element, &mesh, &logical_coords, &element_size,
        &neighbor_sizes
      ](const scalar::type& local_input, const double left, const double right,
        const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes, expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [
        &minmod, &element, &mesh, &logical_coords, &element_size,
        &neighbor_sizes
      ](const scalar::type& local_input, const double left,
        const double right) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes);
  };

  const double muscl_slope_factor =
      (minmod.minmod_type() == SlopeLimiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const auto func = [&muscl_slope_factor](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    // For easier testing, center the quadratic term on the grid: otherwise this
    // term will affect the average slope on the element.
    return 13.0 + 4.0 * muscl_slope_factor * x + 2.5 * square(x);
  };
  const auto input = scalar::type(func(logical_coords));
  const double mean = mean_value(get(input), mesh);

  // Steepness test
  // Because the mesh is higher-than-linear order, the limiter will generally
  // activate due to linearizing the solution, even in cases where slope is OK.
  if (minmod.minmod_type() == SlopeLimiters::MinmodType::LambdaPiN) {
    // However, the LambdaPiN limiter's troubled cell detector will avoid
    // limiting certain smooth solutions; this avoidance will kick in here if,
    // max(u_mean - u_left AND u_right - u_mean) < min(difference of means)
    const double du_left = mean - get(input)[0];
    const double du_right = get(input)[mesh.extents(0) - 1] - mean;
    const double du = std::max(du_left, du_right);
    test_limiter_does_not_activate(input, mean - du - 2.0, mean + du + 2.0);
    test_limiter_does_not_activate(input, mean - du, mean + du);
  }
  test_limiter_activates(input, mean - 5.0, mean + 5.0,
                         4.0 * muscl_slope_factor);
  test_limiter_activates(input, mean - 4.01, mean + 4.01,
                         4.0 * muscl_slope_factor);
  // Cases where slope is too steep and needs to be reduced
  test_limiter_activates(input, mean - 3.99, mean + 3.99,
                         3.99 * muscl_slope_factor);
  test_limiter_activates(input, mean - 1.3, mean + 1.9,
                         1.3 * muscl_slope_factor);

  // Local extremum test
  test_limiter_activates(input, 9.4, 2.3, 0.0);
  test_limiter_activates(input, 14.0, 18.2, 0.0);

  // Oscillation test
  test_limiter_activates(input, 14.0, 2.3, 0.0);
}

void test_limiter_action_with_tvbm_correction(
    const size_t number_of_grid_points,
    const SlopeLimiters::MinmodType& minmod_type) noexcept {
  INFO("Testing TVBM correction...");
  CAPTURE(number_of_grid_points);
  const auto element = make_element<1>();
  const auto mesh = Mesh<1>(number_of_grid_points, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = tnsr::I<double, 1>{2.0};
  const auto neighbor_sizes = make_neighbor_sizes_from_local_size(element_size);

  const auto test_limiter_activates =
      [&element, &mesh, &logical_coords, &element_size, &neighbor_sizes ](
          const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
          const scalar::type& local_input, const double left,
          const double right, const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes, expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [&element, &mesh, &logical_coords, &element_size, &neighbor_sizes ](
          const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
          const scalar::type& local_input, const double left,
          const double right) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes);
  };

  // Make other limiters of same type but with different TBVM constants.
  // Slopes will be compared to m * h^2, where here h = 2
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod_m0(minmod_type);
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod_m1(minmod_type,
                                                               1.0);
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod_m2(minmod_type,
                                                               2.0);

  const auto func = [](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    return 21.6 + 7.2 * get<0>(coords);
  };
  const auto input = scalar::type(func(logical_coords));

  // The TVBM constant sets a threshold slope, below which the solution will not
  // be limited. We test this by increasing the TVBM constant until the limiter
  // stops activating / stops changing the solution.
  const double left =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 8.4 : 15.0;
  const double right =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 34.8 : 30.0;
  test_limiter_activates(minmod_m0, input, left, right, 6.6);
  test_limiter_activates(minmod_m1, input, left, right, 6.6);
  test_limiter_does_not_activate(minmod_m2, input, left, right);
}

// Here we test the coupling of the LambdaPiN troubled cell detector with the
// TVBM constant value.
void test_lambda_pin_troubled_cell_tvbm_correction(
    const size_t number_of_grid_points) noexcept {
  INFO("Testing LambdaPiN-TVBM correction...");
  CAPTURE(number_of_grid_points);
  const auto element = make_element<1>();
  const auto mesh = Mesh<1>(number_of_grid_points, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = tnsr::I<double, 1>{2.0};
  const auto neighbor_sizes = make_neighbor_sizes_from_local_size(element_size);

  const auto test_limiter_activates =
      [&element, &mesh, &logical_coords, &element_size, &neighbor_sizes ](
          const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
          const scalar::type& local_input, const double left,
          const double right, const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes, expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [&element, &mesh, &logical_coords, &element_size, &neighbor_sizes ](
          const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
          const scalar::type& local_input, const double left,
          const double right) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes);
  };

  const auto pi_n = SlopeLimiters::MinmodType::LambdaPiN;
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod_m0(pi_n);
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod_m1(pi_n, 1.0);
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod_m2(pi_n, 2.0);

  const auto func = [](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    return 10.0 * step_function(get<0>(coords));
  };
  const auto input = scalar::type(func(logical_coords));
  const auto input_lin = linearize(get(input), mesh);

  // Establish baseline m = 0 case; LambdaPiN normally avoids limiting when
  // max(edge - mean) <= min(neighbor - mean)
  test_limiter_does_not_activate(minmod_m0, input, 0.0, 10.0);
  test_limiter_does_not_activate(minmod_m0, input, -0.3, 10.2);
  // but does limit if max(edge - mean) > min(neighbor - mean)
  test_limiter_activates(minmod_m0, input, 0.02, 10.0, 4.98);
  test_limiter_activates(minmod_m0, input, 0.0, 9.99, 4.99);

  // However, with a non-zero TVBM constant, LambdaPiN should additionally avoid
  // limiting when max(edge - mean) < TVBM correction.
  // We test first a case where the TVBM correction is too small to affect
  // the limiter action,
  test_limiter_does_not_activate(minmod_m1, input, 0.0, 10.0);
  test_limiter_does_not_activate(minmod_m1, input, -0.3, 10.2);
  test_limiter_activates(minmod_m1, input, 0.02, 10.0, 4.98);
  test_limiter_activates(minmod_m1, input, 0.0, 9.99, 4.99);

  // And a case where the TVBM correction enables LambdaPiN to avoid limiting
  // (Note that the slope here is still too large to avoid limiting through
  // the normal TVBM tolerance.)
  test_limiter_does_not_activate(minmod_m2, input, 0.0, 10.0);
  test_limiter_does_not_activate(minmod_m2, input, -0.3, 10.2);
  test_limiter_does_not_activate(minmod_m2, input, 0.02, 10.0);
  test_limiter_does_not_activate(minmod_m2, input, 0.0, 9.99);
}

void test_limiter_action_at_boundary(
    const size_t number_of_grid_points,
    const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod) noexcept {
  INFO("Testing limiter at boundary...");
  CAPTURE(number_of_grid_points);
  const auto mesh = Mesh<1>(number_of_grid_points, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = tnsr::I<double, 1>{2.0};

  const auto func = [](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    return 1.2 * get<0>(coords);
  };
  const auto input = scalar::type(func(logical_coords));

  // Test with element that has external lower-xi boundary
  const auto element_at_lower_xi_boundary = Element<1>{
      ElementId<1>{0}, Element<1>::Neighbors_t{{Direction<1>::upper_xi(),
                                                make_neighbor_with_id<1>(2)}}};
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_limiter_activates_work(
        minmod, input, {{Direction<1>::upper_xi(), Scalar<double>{neighbor}}},
        element_at_lower_xi_boundary, mesh, logical_coords, element_size,
        {{Direction<1>::upper_xi(), element_size}}, 0.0);
  }

  // Test with element that has external upper-xi boundary
  const auto element_at_upper_xi_boundary = Element<1>{
      ElementId<1>{0}, Element<1>::Neighbors_t{{Direction<1>::lower_xi(),
                                                make_neighbor_with_id<1>(1)}}};
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_limiter_activates_work(
        minmod, input, {{Direction<1>::lower_xi(), Scalar<double>{neighbor}}},
        element_at_upper_xi_boundary, mesh, logical_coords, element_size,
        {{Direction<1>::lower_xi(), element_size}}, 0.0);
  }
}

void test_limiter_action_with_different_size_neighbor(
    const size_t number_of_grid_points,
    const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod) noexcept {
  INFO("Testing limiter with neighboring elements of different size...");
  CAPTURE(number_of_grid_points);
  const auto element = make_element<1>();
  const auto mesh = Mesh<1>(number_of_grid_points, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const double dx = 1.0;
  const auto element_size = tnsr::I<double, 1>{dx};

  const auto test_limiter_activates =
      [&minmod, &element, &mesh, &logical_coords, &element_size ](
          const scalar::type& local_input, const double left,
          const double right,
          const std::unordered_map<Direction<1>, tnsr::I<double, 1>>&
              neighbor_sizes,
          const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes, expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [&minmod, &element, &mesh, &logical_coords, &element_size ](
          const scalar::type& local_input, const double left,
          const double right,
          const std::unordered_map<Direction<1>, tnsr::I<double, 1>>&
              neighbor_sizes) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, make_neighbor_means(left, right), element, mesh,
        logical_coords, element_size, neighbor_sizes);
  };
  const auto make_neighbor_sizes = [](const double left,
                                      const double right) noexcept {
    return std::unordered_map<Direction<1>, tnsr::I<double, 1>>{
        {Direction<1>::lower_xi(), tnsr::I<double, 1>(left)},
        {Direction<1>::upper_xi(), tnsr::I<double, 1>(right)}};
  };

  const double muscl_slope_factor =
      (minmod.minmod_type() == SlopeLimiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  const auto func = [&muscl_slope_factor](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    return 2.0 + 1.2 * muscl_slope_factor * get<0>(coords);
  };
  const auto input = scalar::type(func(logical_coords));

  // Establish baseline using evenly-sized elements
  test_limiter_does_not_activate(
      input, 0.8 - eps, 3.2 + eps,
      make_neighbor_sizes_from_local_size(element_size));

  const auto larger_right = make_neighbor_sizes(dx, 2.0 * dx);
  // Larger neighbor with same mean => true reduction in slope => trigger
  test_limiter_activates(input, 0.8 - eps, 3.2, larger_right,
                         0.8 * muscl_slope_factor);
  // Larger neighbor with larger mean => same slope => no trigger
  test_limiter_does_not_activate(input, 0.8 - eps, 3.8 + eps, larger_right);

  const auto smaller_right = make_neighbor_sizes(dx, 0.5 * dx);
  // Smaller neighbor with same mean => increased slope => no trigger
  test_limiter_does_not_activate(input, 0.8 - eps, 3.2 + eps, smaller_right);
  // Smaller neighbor with lower mean => same slope => no trigger
  test_limiter_does_not_activate(input, 0.8 - eps, 2.9 + eps, smaller_right);

  const auto larger_left = make_neighbor_sizes(2.0 * dx, dx);
  test_limiter_activates(input, 0.8, 3.2 + eps, larger_left,
                         0.8 * muscl_slope_factor);
  test_limiter_does_not_activate(input, 0.2 - eps, 3.2 + eps, larger_left);

  const auto smaller_left = make_neighbor_sizes(0.5 * dx, dx);
  test_limiter_does_not_activate(input, 0.8 - eps, 3.2 + eps, smaller_left);
  test_limiter_does_not_activate(input, 1.1 - eps, 3.2 + eps, smaller_left);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.LambdaPi1.1d_detailed_action_test",
    "[SlopeLimiters][Unit]") {
  // The goal of this test is to check, in detail, the action of the limiter on
  // a wide spectrum of possible inputs.
  //
  // These checks are performed in the various "test_limiter_action_X"
  // functions, which apply the limiter to many different inputs and check that
  // the correct output is received. In addition to verifying that the limiter
  // correctly handles many different smooth or shock-like input states, the
  // following scenarios are also checked:
  // - elements with/without an external boundary
  // - linear and higher-than-linear order meshes
  // - TVBM corrections to the limiter
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod(
      SlopeLimiters::MinmodType::LambdaPi1);

  for (const auto num_grid_points : std::array<size_t, 2>{{2, 4}}) {
    test_limiter_action_on_constant_function(num_grid_points, minmod);
    test_limiter_action_on_linear_function(num_grid_points, minmod);
    test_limiter_action_with_tvbm_correction(num_grid_points,
                                             minmod.minmod_type());
    test_limiter_action_at_boundary(num_grid_points, minmod);
    test_limiter_action_with_different_size_neighbor(num_grid_points, minmod);
  }

  // This test only makes sense on higher-than-linear order meshes
  test_limiter_action_on_quadratic_function(4, minmod);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.LambdaPiN.1d_detailed_action_test",
    "[SlopeLimiters][Unit]") {
  // The goal of this test is to check, in detail, the action of the limiter on
  // a wide spectrum of possible inputs.
  // See the "LambdaPi1.1d_detailed_action_test" test case for more details.
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod(
      SlopeLimiters::MinmodType::LambdaPiN);

  for (const auto num_grid_points : std::array<size_t, 2>{{2, 4}}) {
    test_limiter_action_on_constant_function(num_grid_points, minmod);
    test_limiter_action_on_linear_function(num_grid_points, minmod);
    test_limiter_action_with_tvbm_correction(num_grid_points,
                                             minmod.minmod_type());
    test_limiter_action_at_boundary(num_grid_points, minmod);
    test_limiter_action_with_different_size_neighbor(num_grid_points, minmod);
  }

  // These tests only makes sense on higher-than-linear order meshes
  test_limiter_action_on_quadratic_function(4, minmod);
  test_lambda_pin_troubled_cell_tvbm_correction(4);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.Muscl.1d_detailed_action_test",
    "[SlopeLimiters][Unit]") {
  // The goal of this test is to check, in detail, the action of the limiter on
  // a wide spectrum of possible inputs.
  // See the "LambdaPi1.1d_detailed_action_test" test case for more details.
  const SlopeLimiters::Minmod<1, tmpl::list<scalar>> minmod(
      SlopeLimiters::MinmodType::Muscl);

  for (const auto num_grid_points : std::array<size_t, 2>{{2, 4}}) {
    test_limiter_action_on_constant_function(num_grid_points, minmod);
    test_limiter_action_on_linear_function(num_grid_points, minmod);
    test_limiter_action_with_tvbm_correction(num_grid_points,
                                             minmod.minmod_type());
    test_limiter_action_at_boundary(num_grid_points, minmod);
    test_limiter_action_with_different_size_neighbor(num_grid_points, minmod);
  }

  // This test only makes sense on higher-than-linear order meshes
  test_limiter_action_on_quadratic_function(4, minmod);
}

namespace {
// Helper function for testing Minmod::data_for_neighbors()
template <size_t VolumeDim>
void test_data_for_neighbors_work(
    const Scalar<DataVector>& input_scalar,
    const tnsr::I<DataVector, VolumeDim>& input_vector,
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>&
        logical_coords) noexcept {
  // To streamline the testing of the apply function, the test sets up
  // identical data for all components of input_vector. To better test the
  // data_for_neighbors function, we first modify the input so the data
  // aren't all identical:
  auto modified_vector = input_vector;
  for (size_t d = 0; d < VolumeDim; ++d) {
    modified_vector.get(d) += (d + 1.0) - 2.7 * square(logical_coords.get(d));
  }

  Scalar<double> mean_of_scalar{{{0.0}}};
  tnsr::I<double, VolumeDim> mean_of_vector =
      make_with_value<tnsr::I<double, VolumeDim>>(0.0, 0.0);

  const SlopeLimiters::Minmod<VolumeDim, tmpl::list<scalar, vector<VolumeDim>>>
      minmod(SlopeLimiters::MinmodType::LambdaPi1);
  minmod.data_for_neighbors(make_not_null(&mean_of_scalar),
                            make_not_null(&mean_of_vector), input_scalar,
                            modified_vector, mesh);
  CHECK(get(mean_of_scalar) == approx(mean_value(get(input_scalar), mesh)));
  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK(mean_of_vector.get(d) ==
          approx(mean_value(modified_vector.get(d), mesh)));
  }
}

// Helper function for testing Minmod::apply()
template <size_t VolumeDim>
void test_apply_work(
    const Scalar<DataVector>& input_scalar,
    const std::unordered_map<Direction<VolumeDim>, Scalar<double>>&
        neighbor_scalars,
    const std::array<double, VolumeDim>& target_scalar_slope,
    const tnsr::I<DataVector, VolumeDim>& input_vector,
    const std::unordered_map<Direction<VolumeDim>, tnsr::I<double, VolumeDim>>&
        neighbor_vectors,
    const std::array<std::array<double, VolumeDim>, VolumeDim>&
        target_vector_slope,
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const tnsr::I<double, VolumeDim>& element_size) noexcept {
  auto scalar_to_limit = input_scalar;
  auto vector_to_limit = input_vector;

  const auto element = make_element<VolumeDim>();
  const auto neighbor_sizes = make_neighbor_sizes_from_local_size(element_size);
  const SlopeLimiters::Minmod<VolumeDim, tmpl::list<scalar, vector<VolumeDim>>>
      minmod(SlopeLimiters::MinmodType::LambdaPi1);
  const bool limiter_activated = minmod.apply(
      make_not_null(&scalar_to_limit), make_not_null(&vector_to_limit),
      neighbor_scalars, neighbor_vectors, element, mesh, logical_coords,
      element_size, neighbor_sizes);

  CHECK(limiter_activated);

  CAPTURE(input_scalar);
  CAPTURE(scalar_to_limit);
  CAPTURE(neighbor_scalars);
  CAPTURE(target_scalar_slope);

  const auto expected_limiter_output = [&logical_coords, &mesh ](
      const DataVector& input,
      const std::array<double, VolumeDim> expected_slope) noexcept {
    auto result = make_with_value<DataVector>(input, mean_value(input, mesh));
    for (size_t d = 0; d < VolumeDim; ++d) {
      result += logical_coords.get(d) * gsl::at(expected_slope, d);
    }
    return result;
  };

  CAPTURE(expected_limiter_output(get(input_scalar), target_scalar_slope));

  CHECK_ITERABLE_APPROX(
      get(scalar_to_limit),
      expected_limiter_output(get(input_scalar), target_scalar_slope));
  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK_ITERABLE_APPROX(
        vector_to_limit.get(d),
        expected_limiter_output(input_vector.get(d),
                                gsl::at(target_vector_slope, d)));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.LambdaPi1.1d_pipeline_test",
    "[SlopeLimiters][Unit]") {
  // The goals of this test are,
  // 1. check Minmod::data_for_neighbors
  // 2. check that Minmod::apply limits different tensors independently
  // See comments in the 3D test for full details.
  //
  // a. Generate data to fill all tensor components.
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = tnsr::I<double, 1>{{{0.5}}};
  const auto true_slope = std::array<double, 1>{{2.0}};
  const auto func = [&true_slope](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    return 1.0 + true_slope[0] * x + 3.3 * square(x);
  };
  const auto data = DataVector{func(logical_coords)};
  const double mean = mean_value(data, mesh);
  const auto input_scalar = scalar::type{data};
  const auto input_vector = vector<1>::type{data};

  test_data_for_neighbors_work(input_scalar, input_vector, mesh,
                               logical_coords);

  // b. Generate neighbor data for the scalar and vector Tensors.
  // The scalar we treat as a shock: we want the slope to be reduced
  const auto target_scalar_slope = std::array<double, 1>{{1.2}};
  const auto neighbor_scalars =
      std::unordered_map<Direction<1>, Scalar<double>>{
          {Direction<1>::lower_xi(),
           Scalar<double>(mean - target_scalar_slope[0])},
          {Direction<1>::upper_xi(),
           Scalar<double>(mean + target_scalar_slope[0])},
      };

  // The vector x-component we treat as a smooth function: no limiter action
  const auto target_vector_slope =
      std::array<std::array<double, 1>, 1>{{true_slope}};
  const auto neighbor_vectors =
      std::unordered_map<Direction<1>, tnsr::I<double, 1>>{
          {Direction<1>::lower_xi(),
           tnsr::I<double, 1>{{{mean - 2.0 * true_slope[0]}}}},
          {Direction<1>::upper_xi(),
           tnsr::I<double, 1>{{{mean + 2.0 * true_slope[0]}}}}};

  test_apply_work(input_scalar, neighbor_scalars, target_scalar_slope,
                  input_vector, neighbor_vectors, target_vector_slope, mesh,
                  logical_coords, element_size);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.LambdaPi1.2d_pipeline_test",
    "[SlopeLimiters][Unit]") {
  // The goals of this test are,
  // 1. check Minmod::data_for_neighbors
  // 2. check that Minmod::apply limits different tensors independently
  // 3. check that Minmod::apply limits different dimensions independently
  // See comments in the 3D test for full details.
  //
  // a. Generate data to fill all tensor components.
  const auto mesh =
      Mesh<2>(std::array<size_t, 2>{{3, 3}}, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = tnsr::I<double, 2>{{{0.5, 1.0}}};
  const auto true_slope = std::array<double, 2>{{2.0, -3.0}};
  const auto& func = [&true_slope](
      const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    return 1.0 + true_slope[0] * x + 3.3 * square(x) + true_slope[1] * y +
           square(y);
  };
  const auto data = DataVector{func(logical_coords)};
  const double mean = mean_value(data, mesh);
  const auto input_scalar = scalar::type{data};
  const auto input_vector = vector<2>::type{data};

  test_data_for_neighbors_work(input_scalar, input_vector, mesh,
                               logical_coords);

  // b. Generate neighbor data for the scalar and vector Tensors.
  // The scalar we treat as a 3D shock: we want each slope to be reduced
  const auto target_scalar_slope = std::array<double, 2>{{1.2, -2.2}};
  const auto neighbor_scalar_func = [&mean, &target_scalar_slope](
                                        const size_t dim, const int sign) {
    return Scalar<double>(mean + sign * gsl::at(target_scalar_slope, dim));
  };
  const auto neighbor_scalars =
      std::unordered_map<Direction<2>, Scalar<double>>{
          {Direction<2>::lower_xi(), neighbor_scalar_func(0, -1)},
          {Direction<2>::upper_xi(), neighbor_scalar_func(0, 1)},
          {Direction<2>::lower_eta(), neighbor_scalar_func(1, -1)},
          {Direction<2>::upper_eta(), neighbor_scalar_func(1, 1)}};

  // The vector we treat differently in each component, to check the limiter
  // acts independently on each:
  // - the x-component we treat as a smooth function: no limiter action
  // - the y-component we treat as a shock in y-direction only
  const auto target_vy_slope = std::array<double, 2>{{true_slope[0], -2.2}};
  const auto target_vector_slope =
      std::array<std::array<double, 2>, 2>{{true_slope, target_vy_slope}};
  const auto neighbor_vx_slope =
      std::array<double, 2>{{2.0 * true_slope[0], 2.0 * true_slope[1]}};
  const auto neighbor_vy_slope =
      std::array<double, 2>{{2.0 * true_slope[0], -2.2}};
  const auto neighbor_vector_slope = std::array<std::array<double, 2>, 2>{
      {neighbor_vx_slope, neighbor_vy_slope}};
  const auto neighbor_vector_func = [&mean, &neighbor_vector_slope](
                                        const size_t dim, const int sign) {
    return tnsr::I<double, 2>{
        {{mean + sign * gsl::at(neighbor_vector_slope[0], dim),
          mean + sign * gsl::at(neighbor_vector_slope[1], dim)}}};
  };
  const auto neighbor_vectors =
      std::unordered_map<Direction<2>, tnsr::I<double, 2>>{
          {Direction<2>::lower_xi(), neighbor_vector_func(0, -1)},
          {Direction<2>::upper_xi(), neighbor_vector_func(0, 1)},
          {Direction<2>::lower_eta(), neighbor_vector_func(1, -1)},
          {Direction<2>::upper_eta(), neighbor_vector_func(1, 1)}};

  test_apply_work(input_scalar, neighbor_scalars, target_scalar_slope,
                  input_vector, neighbor_vectors, target_vector_slope, mesh,
                  logical_coords, element_size);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.LambdaPi1.3d_pipeline_test",
    "[SlopeLimiters][Unit]") {
  // The goals of this test are,
  // 1. check Minmod::data_for_neighbors
  // 2. check that Minmod::apply limits different tensors independently
  // 3. check that Minmod::apply limits different dimensions independently
  //
  // The steps taken to meet these goals are:
  // a. set up values in two Tensor<DataVector>s, one scalar and one vector,
  //    then test that Minmod::data_for_neighbors has correct output
  // b. set up neighbor values for these two tensors, then test that
  //    Minmod::apply has correct output
  //
  // These steps are detailed through the test.
  //
  // a. Generate data to fill all tensor components. Note that:
  // - There is no loss of generality from using the same data in every tensor
  //   component, because the neighbor states (and limiter action) will differ.
  // - Quadratic terms are centered on the element so they don't affect the
  //   mean slope on the element.
  const auto mesh =
      Mesh<3>(std::array<size_t, 3>{{3, 3, 4}}, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = tnsr::I<double, 3>{{{0.5, 1.0, 0.8}}};
  const auto true_slope = std::array<double, 3>{{2.0, -3.0, 1.0}};
  const auto func = [&true_slope](
      const tnsr::I<DataVector, 3, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    return 1.0 + true_slope[0] * x + 3.3 * square(x) + true_slope[1] * y +
           square(y) + true_slope[2] * z - square(z);
  };
  const auto data = DataVector{func(logical_coords)};
  const double mean = mean_value(data, mesh);
  const auto input_scalar = scalar::type{data};
  const auto input_vector = vector<3>::type{data};

  test_data_for_neighbors_work(input_scalar, input_vector, mesh,
                               logical_coords);

  // b. Generate neighbor data for the scalar and vector Tensors.
  // The scalar we treat as a 3D shock: we want each slope to be reduced
  const auto target_scalar_slope = std::array<double, 3>{{1.2, -2.2, 0.1}};
  // This function generates the desired neighbor mean value by extrapolating
  // from the local mean value and the desired post-limiting slope
  const auto neighbor_scalar_func = [&mean, &target_scalar_slope](
                                        const size_t dim, const int sign) {
    // The neighbor values are constructed according to
    //   mean_neighbor = mean +/- target_slope * (center_distance / 2.0)
    // which enables easy control of whether the local slope should be reduced
    // by the limiter. This expresion simplifies in logical coordinates,
    // because the center-to-center distance to the neighbor element is 2.0:
    return Scalar<double>(mean + sign * gsl::at(target_scalar_slope, dim));
  };
  const auto neighbor_scalars =
      std::unordered_map<Direction<3>, Scalar<double>>{
          {Direction<3>::lower_xi(), neighbor_scalar_func(0, -1)},
          {Direction<3>::upper_xi(), neighbor_scalar_func(0, 1)},
          {Direction<3>::lower_eta(), neighbor_scalar_func(1, -1)},
          {Direction<3>::upper_eta(), neighbor_scalar_func(1, 1)},
          {Direction<3>::lower_zeta(), neighbor_scalar_func(2, -1)},
          {Direction<3>::upper_zeta(), neighbor_scalar_func(2, 1)}};

  // The vector we treat differently in each component, to verify that the
  // limiter acts independently on each:
  // - the x-component we treat as a smooth function: no limiter action
  // - the y-component we treat as a shock in z-direction only
  // - the z-component we treat as a local maximum, so neighbors < mean
  // For components/directions where we want no limiter action, the desired
  // slope is just the input slope. We actually steepen the slope a little when
  // passing it into neighbor_vector_func, because we want to avoid roundoff
  // errors in comparing slopes.
  const auto target_vy_slope =
      std::array<double, 3>{{true_slope[0], true_slope[1], 0.1}};
  const auto target_vz_slope = std::array<double, 3>{{0.0, 0.0, 0.0}};
  const auto target_vector_slope = std::array<std::array<double, 3>, 3>{
      {true_slope, target_vy_slope, target_vz_slope}};
  const auto neighbor_vx_slope = std::array<double, 3>{
      {2.0 * true_slope[0], 2.0 * true_slope[1], 2.0 * true_slope[2]}};
  const auto neighbor_vy_slope =
      std::array<double, 3>{{2.0 * true_slope[0], 2.0 * true_slope[1], 0.1}};
  const auto neighbor_vector_slope = std::array<std::array<double, 3>, 2>{
      {neighbor_vx_slope, neighbor_vy_slope}};
  // This function generates the desired neighbor mean value
  const auto neighbor_vector_func = [&mean, &neighbor_vector_slope](
                                        const size_t dim, const int sign) {
    return tnsr::I<double, 3>{
        {{mean + sign * gsl::at(neighbor_vector_slope[0], dim),
          mean + sign * gsl::at(neighbor_vector_slope[1], dim),
          mean - 1.1 - dim - sign}}};  // arbitrary, but smaller than mean
  };
  const auto neighbor_vectors =
      std::unordered_map<Direction<3>, tnsr::I<double, 3>>{
          {Direction<3>::lower_xi(), neighbor_vector_func(0, -1)},
          {Direction<3>::upper_xi(), neighbor_vector_func(0, 1)},
          {Direction<3>::lower_eta(), neighbor_vector_func(1, -1)},
          {Direction<3>::upper_eta(), neighbor_vector_func(1, 1)},
          {Direction<3>::lower_zeta(), neighbor_vector_func(2, -1)},
          {Direction<3>::upper_zeta(), neighbor_vector_func(2, 1)}};

  test_apply_work(input_scalar, neighbor_scalars, target_scalar_slope,
                  input_vector, neighbor_vectors, target_vector_slope, mesh,
                  logical_coords, element_size);
}
