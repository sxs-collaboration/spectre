// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
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

// Values of left_id, right_id based on make_element<1> above.
auto make_neighbor_packaged_data(
    const double left, const double right,
    const std::array<double, 1>& local_size_for_neighbor_size) noexcept {
  constexpr size_t left_id = 1;
  constexpr size_t right_id = 2;
  return std::unordered_map<
      std::pair<Direction<1>, ElementId<1>>,
      SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData,
      boost::hash<std::pair<Direction<1>, ElementId<1>>>>{
      std::make_pair(
          std::make_pair(Direction<1>::lower_xi(), ElementId<1>(left_id)),
          SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData{
              Scalar<double>(left), local_size_for_neighbor_size}),
      std::make_pair(
          std::make_pair(Direction<1>::upper_xi(), ElementId<1>(right_id)),
          SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData{
              Scalar<double>(right), local_size_for_neighbor_size})};
}

// Values of left_id, right_id based on make_element<1> above.
auto make_neighbor_packaged_data(const double left, const double right,
                                 const double left_size,
                                 const double right_size) noexcept {
  constexpr size_t left_id = 1;
  constexpr size_t right_id = 2;
  return std::unordered_map<
      std::pair<Direction<1>, ElementId<1>>,
      SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData,
      boost::hash<std::pair<Direction<1>, ElementId<1>>>>{
      std::make_pair(
          std::make_pair(Direction<1>::lower_xi(), ElementId<1>(left_id)),
          SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData{
              Scalar<double>(left), make_array<1>(left_size)}),
      std::make_pair(
          std::make_pair(Direction<1>::upper_xi(), ElementId<1>(right_id)),
          SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData{
              Scalar<double>(right), make_array<1>(right_size)})};
}

// Test that the limiter activates in the x-direction only. Domain quantities
// and input scalar<DataVector> may be of higher dimension VolumeDim.
template <size_t VolumeDim>
void test_limiter_activates_work(
    const SlopeLimiters::Minmod<VolumeDim, tmpl::list<scalar>>& minmod,
    const scalar::type& input, const Element<VolumeDim>& element,
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename SlopeLimiters::Minmod<VolumeDim,
                                       tmpl::list<scalar>>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const double expected_slope) noexcept {
  auto input_to_limit = input;
  const bool limiter_activated =
      minmod(make_not_null(&input_to_limit), element, mesh, logical_coords,
             element_size, neighbor_data);
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
    const scalar::type& input, const Element<1>& element, const Mesh<1>& mesh,
    const tnsr::I<DataVector, 1, Frame::Logical>& logical_coords,
    const std::array<double, 1>& element_size,
    const std::unordered_map<
        std::pair<Direction<1>, ElementId<1>>,
        SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData,
        boost::hash<std::pair<Direction<1>, ElementId<1>>>>&
        neighbor_data) noexcept {
  auto input_to_limit = input;
  const bool limiter_activated =
      minmod(make_not_null(&input_to_limit), element, mesh, logical_coords,
             element_size, neighbor_data);
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
  const auto element_size = make_array<1>(1.2);
  for (const double left : {-0.4, value, 1.2}) {
    for (const double right : {0.2, value, 0.9}) {
      test_limiter_does_not_activate_work(
          minmod, input, element, mesh, logical_coords, element_size,
          make_neighbor_packaged_data(left, right, element_size));
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
  const auto element_size = make_array<1>(2.0);

  const auto test_limiter_activates =
      [&minmod, &element, &mesh, &logical_coords, &element_size ](
          const scalar::type& local_input, const double left,
          const double right, const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, element_size), expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [&minmod, &element, &mesh, &logical_coords, &element_size ](
          const scalar::type& local_input, const double left,
          const double right) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, element_size));
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
  const auto element_size = make_array<1>(2.0);

  const auto test_limiter_activates =
      [&minmod, &element, &mesh, &logical_coords, &element_size ](
          const scalar::type& local_input, const double left,
          const double right, const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, element_size), expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [&minmod, &element, &mesh, &logical_coords, &element_size ](
          const scalar::type& local_input, const double left,
          const double right) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, element_size));
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
  const auto element_size = make_array<1>(2.0);

  const auto test_limiter_activates =
      [&element, &mesh, &logical_coords, &element_size ](
          const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
          const scalar::type& local_input, const double left,
          const double right, const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, element_size), expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [&element, &mesh, &logical_coords, &element_size ](
          const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
          const scalar::type& local_input, const double left,
          const double right) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, element_size));
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
  const auto element_size = make_array<1>(2.0);

  const auto test_limiter_activates =
      [&element, &mesh, &logical_coords, &element_size ](
          const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
          const scalar::type& local_input, const double left,
          const double right, const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, element_size), expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [&element, &mesh, &logical_coords, &element_size ](
          const SlopeLimiters::Minmod<1, tmpl::list<scalar>>& minmod,
          const scalar::type& local_input, const double left,
          const double right) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, element_size));
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
  const auto element_size = make_array<1>(2.0);

  const auto func = [](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    return 1.2 * get<0>(coords);
  };
  const auto input = scalar::type(func(logical_coords));

  // Test with element that has external lower-xi boundary
  // Neighbor on upper-xi side has ElementId == 2
  const auto element_at_lower_xi_boundary = Element<1>{
      ElementId<1>{0}, Element<1>::Neighbors_t{{Direction<1>::upper_xi(),
                                                make_neighbor_with_id<1>(2)}}};
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_limiter_activates_work(
        minmod, input, element_at_lower_xi_boundary, mesh, logical_coords,
        element_size,
        {{std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2)),
          SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData{
              Scalar<double>{neighbor}, element_size}}},
        0.0);
  }

  // Test with element that has external upper-xi boundary
  // Neighbor on lower-xi side has ElementId == 1
  const auto element_at_upper_xi_boundary = Element<1>{
      ElementId<1>{0}, Element<1>::Neighbors_t{{Direction<1>::lower_xi(),
                                                make_neighbor_with_id<1>(1)}}};
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_limiter_activates_work(
        minmod, input, element_at_upper_xi_boundary, mesh, logical_coords,
        element_size,
        {{std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1)),
          SlopeLimiters::Minmod<1, tmpl::list<scalar>>::PackagedData{
              Scalar<double>{neighbor}, element_size}}},
        0.0);
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
  const auto element_size = make_array<1>(dx);

  const auto test_limiter_activates =
      [&minmod, &element, &mesh, &logical_coords, &element_size ](
          const scalar::type& local_input, const double left,
          const double right, const double left_size, const double right_size,
          const double expected_slope) noexcept {
    test_limiter_activates_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, left_size, right_size),
        expected_slope);
  };
  const auto test_limiter_does_not_activate =
      [&minmod, &element, &mesh, &logical_coords, &element_size ](
          const scalar::type& local_input, const double left,
          const double right, const double left_size,
          const double right_size) noexcept {
    test_limiter_does_not_activate_work(
        minmod, local_input, element, mesh, logical_coords, element_size,
        make_neighbor_packaged_data(left, right, left_size, right_size));
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
  test_limiter_does_not_activate(input, 0.8 - eps, 3.2 + eps, dx, dx);

  const double larger = 2.0 * dx;
  const double smaller = 0.5 * dx;

  // Larger neighbor with same mean => true reduction in slope => trigger
  test_limiter_activates(input, 0.8 - eps, 3.2, dx, larger,
                         0.8 * muscl_slope_factor);
  // Larger neighbor with larger mean => same slope => no trigger
  test_limiter_does_not_activate(input, 0.8 - eps, 3.8 + eps, dx, larger);

  // Smaller neighbor with same mean => increased slope => no trigger
  test_limiter_does_not_activate(input, 0.8 - eps, 3.2 + eps, dx, smaller);
  // Smaller neighbor with lower mean => same slope => no trigger
  test_limiter_does_not_activate(input, 0.8 - eps, 2.9 + eps, dx, smaller);

  test_limiter_activates(input, 0.8, 3.2 + eps, larger, dx,
                         0.8 * muscl_slope_factor);
  test_limiter_does_not_activate(input, 0.2 - eps, 3.2 + eps, larger, dx);

  test_limiter_does_not_activate(input, 0.8 - eps, 3.2 + eps, smaller, dx);
  test_limiter_does_not_activate(input, 1.1 - eps, 3.2 + eps, smaller, dx);
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
// Make a 2D element with two neighbors in lower_xi, one neighbor in upper_xi.
// Check that lower_xi data from two neighbors is correctly combined in the
// limiting operation.
void test_limiter_action_two_lower_xi_neighbors() noexcept {
  const auto element = Element<2>{
      ElementId<2>{0},
      Element<2>::Neighbors_t{
          {Direction<2>::lower_xi(),
           {std::unordered_set<ElementId<2>>{ElementId<2>(1), ElementId<2>(7)},
            OrientationMap<2>{}}},
          {Direction<2>::upper_xi(), make_neighbor_with_id<2>(2)}}};
  const auto mesh =
      Mesh<2>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const double dx = 1.0;
  const auto element_size = make_array<2>(dx);

  const auto mean = 2.0;
  const auto func = [&mean](
      const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
    return mean + 1.2 * get<0>(coords);
  };
  const auto input = scalar::type(func(logical_coords));

  const auto make_neighbors = [&dx](const double left1, const double left2,
                                    const double right, const double left1_size,
                                    const double left2_size) noexcept {
    using Pack = SlopeLimiters::Minmod<2, tmpl::list<scalar>>::PackagedData;
    return std::unordered_map<
        std::pair<Direction<2>, ElementId<2>>, Pack,
        boost::hash<std::pair<Direction<2>, ElementId<2>>>>{
        std::make_pair(
            std::make_pair(Direction<2>::lower_xi(), ElementId<2>(1)),
            Pack{Scalar<double>(left1), make_array(left1_size, dx)}),
        std::make_pair(
            std::make_pair(Direction<2>::lower_xi(), ElementId<2>(7)),
            Pack{Scalar<double>(left2), make_array(left2_size, dx)}),
        std::make_pair(
            std::make_pair(Direction<2>::upper_xi(), ElementId<2>(2)),
            Pack{Scalar<double>(right), make_array<2>(dx)}),
    };
  };

  const SlopeLimiters::Minmod<2, tmpl::list<scalar>> minmod(
      SlopeLimiters::MinmodType::LambdaPi1);

  // Make two left neighbors with different mean values
  const auto neighbor_data_two_means =
      make_neighbors(mean - 1.1, mean - 1.0, mean + 1.4, dx, dx);
  // Effective neighbor mean (1.1 + 1.0) / 2.0 => 1.05
  test_limiter_activates_work(minmod, input, element, mesh, logical_coords,
                              element_size, neighbor_data_two_means, 1.05);

  // Make two left neighbors with different means and sizes
  const auto neighbor_data_two_sizes =
      make_neighbors(mean - 1.1, mean - 1.0, mean + 1.4, dx, 0.5 * dx);
  // Effective neighbor mean (1.1 + 1.0) / 2.0 => 1.05
  // Average neighbor size (1.0 + 0.5) / 2.0 => 0.75
  // Effective distance (0.75 + 1.0) / 2.0 => 0.875
  test_limiter_activates_work(minmod, input, element, mesh, logical_coords,
                              element_size, neighbor_data_two_sizes,
                              1.05 / 0.875);
}

// See above, but with 4 upper_xi neighbors. Note that in 2D we are unlikely to
// want domains that have more than 2 neighbors across a face; however, because
// the multi-neighbor averaging is dimension-agnostic, this also can represent a
// 3D test.
void test_limiter_action_four_upper_xi_neighbors() noexcept {
  const auto element = Element<2>{
      ElementId<2>{0},
      Element<2>::Neighbors_t{
          {Direction<2>::lower_xi(), make_neighbor_with_id<2>(1)},
          {Direction<2>::upper_xi(),
           {std::unordered_set<ElementId<2>>{ElementId<2>(2), ElementId<2>(7),
                                             ElementId<2>(8), ElementId<2>(9)},
            OrientationMap<2>{}}},
      }};
  const auto mesh =
      Mesh<2>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const double dx = 1.0;
  const auto element_size = make_array<2>(dx);

  const auto mean = 2.0;
  const auto func = [&mean](
      const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
    return mean + 1.2 * get<0>(coords);
  };
  const auto input = scalar::type(func(logical_coords));

  const auto make_neighbors = [&dx](
      const double left, const double right1, const double right2,
      const double right3, const double right4, const double right1_size,
      const double right2_size, const double right3_size,
      const double right4_size) noexcept {
    using Pack = SlopeLimiters::Minmod<2, tmpl::list<scalar>>::PackagedData;
    return std::unordered_map<
        std::pair<Direction<2>, ElementId<2>>, Pack,
        boost::hash<std::pair<Direction<2>, ElementId<2>>>>{
        std::make_pair(
            std::make_pair(Direction<2>::lower_xi(), ElementId<2>(1)),
            Pack{Scalar<double>(left), make_array<2>(dx)}),
        std::make_pair(
            std::make_pair(Direction<2>::upper_xi(), ElementId<2>(2)),
            Pack{Scalar<double>(right1), make_array(right1_size, dx)}),
        std::make_pair(
            std::make_pair(Direction<2>::upper_xi(), ElementId<2>(7)),
            Pack{Scalar<double>(right2), make_array(right2_size, dx)}),
        std::make_pair(
            std::make_pair(Direction<2>::upper_xi(), ElementId<2>(8)),
            Pack{Scalar<double>(right3), make_array(right3_size, dx)}),
        std::make_pair(
            std::make_pair(Direction<2>::upper_xi(), ElementId<2>(9)),
            Pack{Scalar<double>(right4), make_array(right4_size, dx)}),
    };
  };

  const SlopeLimiters::Minmod<2, tmpl::list<scalar>> minmod(
      SlopeLimiters::MinmodType::LambdaPi1);

  // Make four right neighbors with different mean values
  const auto neighbor_data_two_means =
      make_neighbors(mean - 1.4, mean + 1.0, mean + 1.1, mean - 0.2, mean + 1.8,
                     dx, dx, dx, dx);
  // Effective neighbor mean (1.0 + 1.1 - 0.2 + 1.8) / 4.0 => 0.925
  test_limiter_activates_work(minmod, input, element, mesh, logical_coords,
                              element_size, neighbor_data_two_means, 0.925);

  // Make four right neighbors with different means and sizes
  const auto neighbor_data_two_sizes =
      make_neighbors(mean - 1.4, mean + 1.0, mean + 1.1, mean - 0.2, mean + 1.8,
                     dx, 0.5 * dx, 0.5 * dx, 0.5 * dx);
  // Effective neighbor mean (1.0 + 1.1 - 0.2 + 1.8) / 4.0 => 0.925
  // Average neighbor size (1.0 + 0.5 + 0.5 + 0.5) / 4.0 => 0.625
  // Effective distance (0.625 + 1.0) / 2.0 => 0.8125
  test_limiter_activates_work(minmod, input, element, mesh, logical_coords,
                              element_size, neighbor_data_two_sizes,
                              0.925 / 0.8125);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.LambdaPi1.h_refinement_boundary",
    "[SlopeLimiters][Unit]") {
  test_limiter_action_two_lower_xi_neighbors();
  test_limiter_action_four_upper_xi_neighbors();
}

namespace {
// Helper function for testing Minmod::package_data()
template <size_t VolumeDim>
void test_package_data_work(
    const Scalar<DataVector>& input_scalar,
    const tnsr::I<DataVector, VolumeDim>& input_vector,
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) noexcept {
  // To streamline the testing of the op() function, the test sets up
  // identical data for all components of input_vector. To better test the
  // package_data function, we first modify the input so the data
  // aren't all identical:
  auto modified_vector = input_vector;
  for (size_t d = 0; d < VolumeDim; ++d) {
    modified_vector.get(d) += (d + 1.0) - 2.7 * square(logical_coords.get(d));
  }

  const SlopeLimiters::Minmod<VolumeDim, tmpl::list<scalar, vector<VolumeDim>>>
      minmod(SlopeLimiters::MinmodType::LambdaPi1);
  typename SlopeLimiters::Minmod<
      VolumeDim, tmpl::list<scalar, vector<VolumeDim>>>::PackagedData
      packaged_data{};

  // First we test package_data with an identity orientation_map
  minmod.package_data(make_not_null(&packaged_data), input_scalar,
                      modified_vector, mesh, element_size, {});

  // Should not normally look inside the package, but we do so here for testing.
  double lhs =
      get(get<Minmod_detail::to_tensor_double<scalar>>(packaged_data.means_));
  CHECK(lhs == approx(mean_value(get(input_scalar), mesh)));
  for (size_t d = 0; d < VolumeDim; ++d) {
    lhs = get<Minmod_detail::to_tensor_double<vector<VolumeDim>>>(
              packaged_data.means_)
              .get(d);
    CHECK(lhs == approx(mean_value(modified_vector.get(d), mesh)));
  }
  CHECK(packaged_data.element_size_ == element_size);

  // Then we test with a reorientation, as if sending the data to another Block
  minmod.package_data(make_not_null(&packaged_data), input_scalar,
                      modified_vector, mesh, element_size, orientation_map);
  lhs = get(get<Minmod_detail::to_tensor_double<scalar>>(packaged_data.means_));
  CHECK(lhs == approx(mean_value(get(input_scalar), mesh)));
  for (size_t d = 0; d < VolumeDim; ++d) {
    lhs = get<Minmod_detail::to_tensor_double<vector<VolumeDim>>>(
              packaged_data.means_)
              .get(d);
    CHECK(lhs == approx(mean_value(modified_vector.get(d), mesh)));
  }
  CHECK(packaged_data.element_size_ ==
        orientation_map.permute_from_neighbor(element_size));
}

// Helper function for testing Minmod::op()
template <size_t VolumeDim>
void test_work(
    const Scalar<DataVector>& input_scalar,
    const tnsr::I<DataVector, VolumeDim>& input_vector,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename SlopeLimiters::Minmod<
            VolumeDim, tmpl::list<scalar, vector<VolumeDim>>>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::array<double, VolumeDim>& target_scalar_slope,
    const std::array<std::array<double, VolumeDim>, VolumeDim>&
        target_vector_slope) noexcept {
  auto scalar_to_limit = input_scalar;
  auto vector_to_limit = input_vector;

  const auto element = make_element<VolumeDim>();
  const SlopeLimiters::Minmod<VolumeDim, tmpl::list<scalar, vector<VolumeDim>>>
      minmod(SlopeLimiters::MinmodType::LambdaPi1);
  const bool limiter_activated =
      minmod(make_not_null(&scalar_to_limit), make_not_null(&vector_to_limit),
             element, mesh, logical_coords, element_size, neighbor_data);

  CHECK(limiter_activated);

  const auto expected_limiter_output = [&logical_coords, &mesh ](
      const DataVector& input,
      const std::array<double, VolumeDim> expected_slope) noexcept {
    auto result = make_with_value<DataVector>(input, mean_value(input, mesh));
    for (size_t d = 0; d < VolumeDim; ++d) {
      result += logical_coords.get(d) * gsl::at(expected_slope, d);
    }
    return result;
  };

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
  // 1. check Minmod::package_data
  // 2. check that Minmod::op() limits different tensors independently
  // See comments in the 3D test for full details.
  //
  // a. Generate data to fill all tensor components.
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(0.5);
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

  const OrientationMap<1> test_reorientation(
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
  test_package_data_work(input_scalar, input_vector, mesh, logical_coords,
                         element_size, test_reorientation);

  // b. Generate neighbor data for the scalar and vector Tensors.
  std::unordered_map<
      std::pair<Direction<1>, ElementId<1>>,
      SlopeLimiters::Minmod<1, tmpl::list<scalar, vector<1>>>::PackagedData,
      boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<1>, ElementId<1>>, 2> dir_keys = {
      {{Direction<1>::lower_xi(), ElementId<1>(1)},
       {Direction<1>::upper_xi(), ElementId<1>(2)}}};
  neighbor_data[dir_keys[0]].element_size_ = element_size;
  neighbor_data[dir_keys[1]].element_size_ = element_size;

  // The scalar we treat as a shock: we want the slope to be reduced
  const auto target_scalar_slope = std::array<double, 1>{{1.2}};
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[0]].means_) =
      Scalar<double>(mean - target_scalar_slope[0]);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[1]].means_) =
      Scalar<double>(mean + target_scalar_slope[0]);

  // The vector x-component we treat as a smooth function: no limiter action
  const auto target_vector_slope =
      std::array<std::array<double, 1>, 1>{{true_slope}};
  get<Minmod_detail::to_tensor_double<vector<1>>>(
      neighbor_data[dir_keys[0]].means_) =
      tnsr::I<double, 1>(mean - 2.0 * true_slope[0]);
  get<Minmod_detail::to_tensor_double<vector<1>>>(
      neighbor_data[dir_keys[1]].means_) =
      tnsr::I<double, 1>(mean + 2.0 * true_slope[0]);

  test_work(input_scalar, input_vector, neighbor_data, mesh, logical_coords,
            element_size, target_scalar_slope, target_vector_slope);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.LambdaPi1.2d_pipeline_test",
    "[SlopeLimiters][Unit]") {
  // The goals of this test are,
  // 1. check Minmod::package_data
  // 2. check that Minmod::op() limits different tensors independently
  // 3. check that Minmod::op() limits different dimensions independently
  // See comments in the 3D test for full details.
  //
  // a. Generate data to fill all tensor components.
  const auto mesh =
      Mesh<2>(std::array<size_t, 2>{{3, 3}}, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array(0.5, 1.0);
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

  const OrientationMap<2> test_reorientation(std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}});
  test_package_data_work(input_scalar, input_vector, mesh, logical_coords,
                         element_size, test_reorientation);

  // b. Generate neighbor data for the scalar and vector Tensors.
  std::unordered_map<
      std::pair<Direction<2>, ElementId<2>>,
      SlopeLimiters::Minmod<2, tmpl::list<scalar, vector<2>>>::PackagedData,
      boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<2>, ElementId<2>>, 4> dir_keys = {
      {{Direction<2>::lower_xi(), ElementId<2>(1)},
       {Direction<2>::upper_xi(), ElementId<2>(2)},
       {Direction<2>::lower_eta(), ElementId<2>(3)},
       {Direction<2>::upper_eta(), ElementId<2>(4)}}};
  neighbor_data[dir_keys[0]].element_size_ = element_size;
  neighbor_data[dir_keys[1]].element_size_ = element_size;
  neighbor_data[dir_keys[2]].element_size_ = element_size;
  neighbor_data[dir_keys[3]].element_size_ = element_size;

  // The scalar we treat as a 3D shock: we want each slope to be reduced
  const auto target_scalar_slope = std::array<double, 2>{{1.2, -2.2}};
  const auto neighbor_scalar_func = [&mean, &target_scalar_slope](
                                        const size_t dim, const int sign) {
    return Scalar<double>(mean + sign * gsl::at(target_scalar_slope, dim));
  };
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[0]].means_) = neighbor_scalar_func(0, -1);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[1]].means_) = neighbor_scalar_func(0, 1);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[2]].means_) = neighbor_scalar_func(1, -1);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[3]].means_) = neighbor_scalar_func(1, 1);

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
  get<Minmod_detail::to_tensor_double<vector<2>>>(
      neighbor_data[dir_keys[0]].means_) = neighbor_vector_func(0, -1);
  get<Minmod_detail::to_tensor_double<vector<2>>>(
      neighbor_data[dir_keys[1]].means_) = neighbor_vector_func(0, 1);
  get<Minmod_detail::to_tensor_double<vector<2>>>(
      neighbor_data[dir_keys[2]].means_) = neighbor_vector_func(1, -1);
  get<Minmod_detail::to_tensor_double<vector<2>>>(
      neighbor_data[dir_keys[3]].means_) = neighbor_vector_func(1, 1);

  test_work(input_scalar, input_vector, neighbor_data, mesh, logical_coords,
            element_size, target_scalar_slope, target_vector_slope);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.Minmod.LambdaPi1.3d_pipeline_test",
    "[SlopeLimiters][Unit]") {
  // The goals of this test are,
  // 1. check Minmod::package_data
  // 2. check that Minmod::op() limits different tensors independently
  // 3. check that Minmod::op() limits different dimensions independently
  //
  // The steps taken to meet these goals are:
  // a. set up values in two Tensor<DataVector>s, one scalar and one vector,
  //    then test that Minmod::package_data has correct output
  // b. set up neighbor values for these two tensors, then test that
  //    Minmod::op() has correct output
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
  const auto element_size = make_array(0.5, 1.0, 0.8);
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

  const OrientationMap<3> test_reorientation(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::lower_zeta()}});
  test_package_data_work(input_scalar, input_vector, mesh, logical_coords,
                         element_size, test_reorientation);

  // b. Generate neighbor data for the scalar and vector Tensors.
  std::unordered_map<
      std::pair<Direction<3>, ElementId<3>>,
      SlopeLimiters::Minmod<3, tmpl::list<scalar, vector<3>>>::PackagedData,
      boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<3>, ElementId<3>>, 6> dir_keys = {
      {{Direction<3>::lower_xi(), ElementId<3>(1)},
       {Direction<3>::upper_xi(), ElementId<3>(2)},
       {Direction<3>::lower_eta(), ElementId<3>(3)},
       {Direction<3>::upper_eta(), ElementId<3>(4)},
       {Direction<3>::lower_zeta(), ElementId<3>(5)},
       {Direction<3>::upper_zeta(), ElementId<3>(6)}}};
  for (const auto& id_pair : dir_keys) {
    neighbor_data[id_pair].element_size_ = element_size;
  }

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
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[0]].means_) = neighbor_scalar_func(0, -1);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[1]].means_) = neighbor_scalar_func(0, 1);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[2]].means_) = neighbor_scalar_func(1, -1);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[3]].means_) = neighbor_scalar_func(1, 1);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[4]].means_) = neighbor_scalar_func(2, -1);
  get<Minmod_detail::to_tensor_double<scalar>>(
      neighbor_data[dir_keys[5]].means_) = neighbor_scalar_func(2, 1);

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
  get<Minmod_detail::to_tensor_double<vector<3>>>(
      neighbor_data[dir_keys[0]].means_) = neighbor_vector_func(0, -1);
  get<Minmod_detail::to_tensor_double<vector<3>>>(
      neighbor_data[dir_keys[1]].means_) = neighbor_vector_func(0, 1);
  get<Minmod_detail::to_tensor_double<vector<3>>>(
      neighbor_data[dir_keys[2]].means_) = neighbor_vector_func(1, -1);
  get<Minmod_detail::to_tensor_double<vector<3>>>(
      neighbor_data[dir_keys[3]].means_) = neighbor_vector_func(1, 1);
  get<Minmod_detail::to_tensor_double<vector<3>>>(
      neighbor_data[dir_keys[4]].means_) = neighbor_vector_func(2, -1);
  get<Minmod_detail::to_tensor_double<vector<3>>>(
      neighbor_data[dir_keys[5]].means_) = neighbor_vector_func(2, 1);

  test_work(input_scalar, input_vector, neighbor_data, mesh, logical_coords,
            element_size, target_scalar_slope, target_vector_slope);
}
