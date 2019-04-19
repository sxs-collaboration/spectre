// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// Helper function to wrap the allocation of the optimization buffers for the
// troubled cell indicator function.
template <size_t VolumeDim>
bool wrap_allocations_and_tci(
    const gsl::not_null<double*> u_mean,
    const gsl::not_null<std::array<double, VolumeDim>*> u_limited_slopes,
    const SlopeLimiters::MinmodType& minmod_type, const double tvbm_constant,
    const DataVector& u, const Element<VolumeDim>& element,
    const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes) noexcept {
  // Allocate the various temporary buffers.
  DataVector u_lin_buffer(mesh.number_of_grid_points());
  std::array<DataVector, VolumeDim> boundary_buffer{};
  for (size_t d = 0; d < VolumeDim; ++d) {
    const size_t num_points = mesh.slice_away(d).number_of_grid_points();
    gsl::at(boundary_buffer, d) = DataVector(num_points);
  }

  const auto volume_and_slice_buffer_and_indices =
      volume_and_slice_indices(mesh.extents());
  const auto& volume_and_slice_indices =
      volume_and_slice_buffer_and_indices.second;

  return SlopeLimiters::Minmod_detail::troubled_cell_indicator(
      u_mean, u_limited_slopes, make_not_null(&u_lin_buffer),
      make_not_null(&boundary_buffer), minmod_type, tvbm_constant, u, element,
      mesh, element_size, effective_neighbor_means, effective_neighbor_sizes,
      volume_and_slice_indices);
}

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

auto make_two_neighbors(const double left, const double right) noexcept {
  DirectionMap<1, double> result;
  result[Direction<1>::lower_xi()] = left;
  result[Direction<1>::upper_xi()] = right;
  return result;
}

auto make_four_neighbors(const std::array<double, 4>& values) noexcept {
  DirectionMap<2, double> result;
  result[Direction<2>::lower_xi()] = values[0];
  result[Direction<2>::upper_xi()] = values[1];
  result[Direction<2>::lower_eta()] = values[2];
  result[Direction<2>::upper_eta()] = values[3];
  return result;
}

auto make_six_neighbors(const std::array<double, 6>& values) noexcept {
  DirectionMap<3, double> result;
  result[Direction<3>::lower_xi()] = values[0];
  result[Direction<3>::upper_xi()] = values[1];
  result[Direction<3>::lower_eta()] = values[2];
  result[Direction<3>::upper_eta()] = values[3];
  result[Direction<3>::lower_zeta()] = values[4];
  result[Direction<3>::upper_zeta()] = values[5];
  return result;
}

// Test that TCI detects a troubled cell when expected, and returns the correct
// mean and reduced slopes.
template <size_t VolumeDim>
void test_minmod_tci_activates(
    const SlopeLimiters::MinmodType& minmod_type, const double tvbm_constant,
    const DataVector& input, const Element<VolumeDim>& element,
    const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes,
    const std::array<double, VolumeDim>& expected_slopes) noexcept {
  double u_mean{};
  std::array<double, VolumeDim> u_limited_slopes{};
  const bool tci_activated = wrap_allocations_and_tci(
      make_not_null(&u_mean), make_not_null(&u_limited_slopes), minmod_type,
      tvbm_constant, input, element, mesh, element_size,
      effective_neighbor_means, effective_neighbor_sizes);
  CHECK(tci_activated);
  CHECK(u_mean == approx(mean_value(input, mesh)));
  CHECK_ITERABLE_APPROX(u_limited_slopes, expected_slopes);
}

// Test that TCI does not detect a troubled cell.
template <size_t VolumeDim>
void test_minmod_tci_does_not_activate(
    const SlopeLimiters::MinmodType& minmod_type, const double tvbm_constant,
    const DataVector& input, const Element<VolumeDim>& element,
    const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes,
    const std::array<double, VolumeDim>& original_slopes) noexcept {
  double u_mean{};
  std::array<double, VolumeDim> u_limited_slopes{};
  const bool tci_activated = wrap_allocations_and_tci(
      make_not_null(&u_mean), make_not_null(&u_limited_slopes), minmod_type,
      tvbm_constant, input, element, mesh, element_size,
      effective_neighbor_means, effective_neighbor_sizes);
  CHECK_FALSE(tci_activated);
  if (minmod_type != SlopeLimiters::MinmodType::LambdaPiN) {
    CHECK(u_mean == approx(mean_value(input, mesh)));
    CHECK_ITERABLE_APPROX(u_limited_slopes, original_slopes);
  }
}

void test_tci_on_linear_function(
    const size_t number_of_grid_points,
    const SlopeLimiters::MinmodType& minmod_type) noexcept {
  INFO("Testing linear function...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const double tvbm_constant = 0.0;
  const auto element = make_element<1>();
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<1>(2.0);

  const auto test_activates =
      [&minmod_type, &tvbm_constant, &element, &mesh, &element_size ](
          const DataVector& local_input, const double left, const double right,
          const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_tci_activates(minmod_type, tvbm_constant, local_input, element,
                              mesh, element_size,
                              make_two_neighbors(left, right),
                              make_two_neighbors(2.0, 2.0), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &tvbm_constant, &element, &mesh, &element_size ](
          const DataVector& local_input, const double left, const double right,
          const double original_slope) noexcept {
    const auto original_slopes = make_array<1>(original_slope);
    test_minmod_tci_does_not_activate(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_two_neighbors(left, right), make_two_neighbors(2.0, 2.0),
        original_slopes);
  };

  // With a MUSCL limiter, the largest allowed slope is half as big as for a
  // LambdaPi1 or LambdaPiN limiter. We can re-use the same test cases by
  // correspondingly scaling the slope:
  const double muscl_slope_factor =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  // Test a positive-slope function
  {
    const auto input = [&muscl_slope_factor, &mesh ]() noexcept {
      const auto coords = logical_coordinates(mesh);
      return DataVector{3.6 + 1.2 * muscl_slope_factor * get<0>(coords)};
    }
    ();

    // Steepness test
    // Limiter does not reduce slope if (difference of means) > (local slope),
    // but reduces slope if (difference of means) < (local slope)
    test_does_not_activate(input, 2.0, 6.0, 1.2 * muscl_slope_factor);
    test_does_not_activate(input, 2.4 - eps, 4.8 + eps,
                           1.2 * muscl_slope_factor);
    test_activates(input, 2.6, 6.0, 1.0 * muscl_slope_factor);
    test_activates(input, 2.0, 4.0, 0.4 * muscl_slope_factor);
    test_activates(input, 2.6, 4.0, 0.4 * muscl_slope_factor);

    // Local extremum test
    // Limiter flattens slope if both neighbors are above (below) the mean
    test_activates(input, 1.0, 2.0, 0.0);
    test_activates(input, 6.0, 9.0, 0.0);

    // Oscillation test
    // Limiter flattens slope if sign(difference of means) != sign(local slope)
    test_activates(input, 3.9, 2.7, 0.0);
  }

  // Test a negative-slope function
  {
    const auto input = [&muscl_slope_factor, &mesh ]() noexcept {
      const auto coords = logical_coordinates(mesh);
      return DataVector{-0.4 - 0.8 * muscl_slope_factor * get<0>(coords)};
    }
    ();

    // Steepness test
    test_does_not_activate(input, 0.9, -2.3, -0.8 * muscl_slope_factor);
    test_does_not_activate(input, 0.4 + eps, -1.2 - eps,
                           -0.8 * muscl_slope_factor);
    test_activates(input, 0.2, -1.2, -0.6 * muscl_slope_factor);
    test_activates(input, 0.4, -0.5, -0.1 * muscl_slope_factor);
    test_activates(input, 0.2, -0.5, -0.1 * muscl_slope_factor);

    // Local extremum test
    test_activates(input, 1.3, -0.1, 0.0);
    test_activates(input, -3.2, -0.8, 0.0);

    // Oscillation test
    test_activates(input, -2.3, 0.2, 0.0);
  }
}

void test_tci_on_quadratic_function(
    const size_t number_of_grid_points,
    const SlopeLimiters::MinmodType& minmod_type) noexcept {
  INFO("Testing quadratic function...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const double tvbm_constant = 0.0;
  const auto element = make_element<1>();
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<1>(2.0);

  const auto test_activates =
      [&minmod_type, &tvbm_constant, &element, &mesh, &element_size ](
          const DataVector& local_input, const double left, const double right,
          const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_tci_activates(minmod_type, tvbm_constant, local_input, element,
                              mesh, element_size,
                              make_two_neighbors(left, right),
                              make_two_neighbors(2.0, 2.0), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &tvbm_constant, &element, &mesh, &element_size ](
          const DataVector& local_input, const double left, const double right,
          const double original_slope) noexcept {
    const auto original_slopes = make_array<1>(original_slope);
    test_minmod_tci_does_not_activate(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_two_neighbors(left, right), make_two_neighbors(2.0, 2.0),
        original_slopes);
  };

  const double muscl_slope_factor =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 0.5 : 1.0;

  const auto input = [&muscl_slope_factor, &mesh ]() noexcept {
    const auto coords = logical_coordinates(mesh);
    const auto& x = get<0>(coords);
    // For easier testing, center the quadratic term on the grid: otherwise this
    // term will affect the average slope on the element.
    return DataVector{13.0 + 4.0 * muscl_slope_factor * x + 2.5 * square(x)};
  }
  ();
  const double mean = mean_value(input, mesh);

  // Steepness test
  // Cases where slope is not steep enough to need limiting
  test_does_not_activate(input, mean - 5.0, mean + 5.0,
                         4.0 * muscl_slope_factor);
  test_does_not_activate(input, mean - 4.01, mean + 4.01,
                         4.0 * muscl_slope_factor);
  // Cases where slope is too steep and needs to be reduced
  test_activates(input, mean - 3.99, mean + 3.99, 3.99 * muscl_slope_factor);
  test_activates(input, mean - 1.3, mean + 1.9, 1.3 * muscl_slope_factor);

  // Local extremum test
  test_activates(input, 9.4, 2.3, 0.0);
  test_activates(input, 14.0, 18.2, 0.0);

  // Oscillation test
  test_activates(input, 14.0, 2.3, 0.0);
}

void test_tci_with_tvbm_correction(
    const size_t number_of_grid_points,
    const SlopeLimiters::MinmodType& minmod_type) noexcept {
  INFO("Testing TVBM correction...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const auto element = make_element<1>();
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<1>(2.0);

  const auto test_activates = [&minmod_type, &element, &mesh, &element_size ](
      const double tvbm_constant, const DataVector& local_input,
      const double left, const double right,
      const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_tci_activates(minmod_type, tvbm_constant, local_input, element,
                              mesh, element_size,
                              make_two_neighbors(left, right),
                              make_two_neighbors(2.0, 2.0), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &element, &mesh, &
       element_size ](const double tvbm_constant, const DataVector& local_input,
                      const double left, const double right,
                      const double original_slope) noexcept {
    const auto original_slopes = make_array<1>(original_slope);
    test_minmod_tci_does_not_activate(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_two_neighbors(left, right), make_two_neighbors(2.0, 2.0),
        original_slopes);
  };

  // Slopes will be compared to m * h^2, where here h = 2
  const double tvbm_m0 = 0.0;
  const double tvbm_m1 = 1.0;
  const double tvbm_m2 = 2.0;

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{21.6 + 7.2 * get<0>(coords)};
  }
  ();

  // The TVBM constant sets a threshold slope, below which the solution will not
  // be limited. We test this by increasing the TVBM constant until the limiter
  // stops activating / stops changing the solution.
  const double left =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 8.4 : 15.0;
  const double right =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 34.8 : 30.0;
  test_activates(tvbm_m0, input, left, right, 6.6);
  test_activates(tvbm_m1, input, left, right, 6.6);
  test_does_not_activate(tvbm_m2, input, left, right, 7.2);
}

// Here we test the coupling of the LambdaPiN troubled cell detector with the
// TVBM constant value.
void test_lambda_pin_troubled_cell_tvbm_correction(
    const size_t number_of_grid_points) noexcept {
  INFO("Testing LambdaPiN-TVBM correction...");
  CAPTURE(number_of_grid_points);
  const auto element = make_element<1>();
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(2.0);

  const auto test_activates = [&element, &mesh, &element_size ](
      const double tvbm_constant, const DataVector& local_input,
      const double left, const double right,
      const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_tci_activates(SlopeLimiters::MinmodType::LambdaPiN,
                              tvbm_constant, local_input, element, mesh,
                              element_size, make_two_neighbors(left, right),
                              make_two_neighbors(2.0, 2.0), expected_slopes);
  };
  const auto test_does_not_activate = [&element, &mesh, &element_size ](
      const double tvbm_constant, const DataVector& local_input,
      const double left, const double right) noexcept {
    // Because in this test the TCI is LambdaPiN, no slopes are returned, and
    // no comparison is made. So set these to NaN
    const auto original_slopes =
        make_array<1>(std::numeric_limits<double>::signaling_NaN());
    test_minmod_tci_does_not_activate(
        SlopeLimiters::MinmodType::LambdaPiN, tvbm_constant, local_input,
        element, mesh, element_size, make_two_neighbors(left, right),
        make_two_neighbors(2.0, 2.0), original_slopes);
  };

  const double m0 = 0.0;
  const double m1 = 1.0;
  const double m2 = 2.0;

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{10.0 * step_function(get<0>(coords))};
  }
  ();

  // Establish baseline m = 0 case; LambdaPiN normally avoids limiting when
  // max(edge - mean) <= min(neighbor - mean)
  test_does_not_activate(m0, input, 0.0, 10.0);
  test_does_not_activate(m0, input, -0.3, 10.2);
  // but does limit if max(edge - mean) > min(neighbor - mean)
  test_activates(m0, input, 0.02, 10.0, 4.98);
  test_activates(m0, input, 0.0, 9.99, 4.99);

  // However, with a non-zero TVBM constant, LambdaPiN should additionally avoid
  // limiting when max(edge - mean) < TVBM correction.
  // We test first a case where the TVBM correction is too small to affect
  // the limiter action,
  test_does_not_activate(m1, input, 0.0, 10.0);
  test_does_not_activate(m1, input, -0.3, 10.2);
  test_activates(m1, input, 0.02, 10.0, 4.98);
  test_activates(m1, input, 0.0, 9.99, 4.99);

  // And a case where the TVBM correction enables LambdaPiN to avoid limiting
  // (Note that the slope here is still too large to avoid limiting through
  // the normal TVBM tolerance.)
  test_does_not_activate(m2, input, 0.0, 10.0);
  test_does_not_activate(m2, input, -0.3, 10.2);
  test_does_not_activate(m2, input, 0.02, 10.0);
  test_does_not_activate(m2, input, 0.0, 9.99);
}

void test_tci_at_boundary(
    const size_t number_of_grid_points,
    const SlopeLimiters::MinmodType& minmod_type) noexcept {
  INFO("Testing limiter at boundary...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const double tvbm_constant = 0.0;
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<1>(2.0);

  const double muscl_slope_factor =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const auto input = [&muscl_slope_factor, &mesh ]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{1.2 * muscl_slope_factor * get<0>(coords)};
  }
  ();

  // Test with element that has external lower-xi boundary
  // Neighbor on upper-xi side has ElementId == 2
  const auto element_at_lower_xi_boundary = Element<1>{
      ElementId<1>{0}, Element<1>::Neighbors_t{{Direction<1>::upper_xi(),
                                                make_neighbor_with_id<1>(2)}}};
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_minmod_tci_activates(
        minmod_type, tvbm_constant, input, element_at_lower_xi_boundary, mesh,
        element_size, {{std::make_pair(Direction<1>::upper_xi(), neighbor)}},
        {{std::make_pair(Direction<1>::upper_xi(), element_size[0])}},
        make_array<1>(0.0));
  }

  // Test with element that has external upper-xi boundary
  // Neighbor on lower-xi side has ElementId == 1
  const auto element_at_upper_xi_boundary = Element<1>{
      ElementId<1>{0}, Element<1>::Neighbors_t{{Direction<1>::lower_xi(),
                                                make_neighbor_with_id<1>(1)}}};
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_minmod_tci_activates(
        minmod_type, tvbm_constant, input, element_at_upper_xi_boundary, mesh,
        element_size, {{std::make_pair(Direction<1>::lower_xi(), neighbor)}},
        {{std::make_pair(Direction<1>::lower_xi(), element_size[0])}},
        make_array<1>(0.0));
  }
}

void test_tci_with_different_size_neighbor(
    const size_t number_of_grid_points,
    const SlopeLimiters::MinmodType& minmod_type) noexcept {
  INFO("Testing limiter with neighboring elements of different size...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const double tvbm_constant = 0.0;
  const auto element = make_element<1>();
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const double dx = 1.0;
  const auto element_size = make_array<1>(dx);

  const auto test_activates =
      [&minmod_type, &tvbm_constant, &element, &mesh, &element_size ](
          const DataVector& local_input, const double left, const double right,
          const double left_size, const double right_size,
          const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_tci_activates(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_two_neighbors(left, right),
        make_two_neighbors(left_size, right_size), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &tvbm_constant, &element, &mesh, &element_size ](
          const DataVector& local_input, const double left, const double right,
          const double left_size, const double right_size,
          const double original_slope) noexcept {
    const auto original_slopes = make_array<1>(original_slope);
    test_minmod_tci_does_not_activate(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_two_neighbors(left, right),
        make_two_neighbors(left_size, right_size), original_slopes);
  };

  const double muscl_slope_factor =
      (minmod_type == SlopeLimiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  const auto input = [&muscl_slope_factor, &mesh ]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{2.0 + 1.2 * muscl_slope_factor * get<0>(coords)};
  }
  ();

  // Establish baseline using evenly-sized elements
  test_does_not_activate(input, 0.8 - eps, 3.2 + eps, dx, dx,
                         1.2 * muscl_slope_factor);

  const double larger = 2.0 * dx;
  const double smaller = 0.5 * dx;

  // Larger neighbor with same mean => true reduction in slope => trigger
  test_activates(input, 0.8 - eps, 3.2, dx, larger, 0.8 * muscl_slope_factor);
  // Larger neighbor with larger mean => same slope => no trigger
  test_does_not_activate(input, 0.8 - eps, 3.8 + eps, dx, larger,
                         1.2 * muscl_slope_factor);

  // Smaller neighbor with same mean => increased slope => no trigger
  test_does_not_activate(input, 0.8 - eps, 3.2 + eps, dx, smaller,
                         1.2 * muscl_slope_factor);
  // Smaller neighbor with lower mean => same slope => no trigger
  test_does_not_activate(input, 0.8 - eps, 2.9 + eps, dx, smaller,
                         1.2 * muscl_slope_factor);

  test_activates(input, 0.8, 3.2 + eps, larger, dx, 0.8 * muscl_slope_factor);
  test_does_not_activate(input, 0.2 - eps, 3.2 + eps, larger, dx,
                         1.2 * muscl_slope_factor);

  test_does_not_activate(input, 0.8 - eps, 3.2 + eps, smaller, dx,
                         1.2 * muscl_slope_factor);
  test_does_not_activate(input, 1.1 - eps, 3.2 + eps, smaller, dx,
                         1.2 * muscl_slope_factor);
}

// In 1D, test combinations of MinmodType, TVBM constant, polynomial order, etc.
// Check that each combination has the expected TCI behavior.
void test_minmod_tci_1d() noexcept {
  INFO("Testing MinmodTci in 1D");
  for (const auto& minmod_type : {SlopeLimiters::MinmodType::LambdaPi1,
                                  SlopeLimiters::MinmodType::LambdaPiN,
                                  SlopeLimiters::MinmodType::Muscl}) {
    for (const auto num_grid_points : std::array<size_t, 2>{{2, 4}}) {
      test_tci_on_linear_function(num_grid_points, minmod_type);
      test_tci_with_tvbm_correction(num_grid_points, minmod_type);
      test_tci_at_boundary(num_grid_points, minmod_type);
      test_tci_with_different_size_neighbor(num_grid_points, minmod_type);
    }
    // This test only makes sense with more than 2 grid points
    test_tci_on_quadratic_function(4, minmod_type);
  }
  // This test only makes sense with LambdaPiN and more than 2 grid points
  test_lambda_pin_troubled_cell_tvbm_correction(4);
}

// In 2D, test that the dimension-by-dimension application of the TCI works as
// expected.
void test_minmod_tci_2d() noexcept {
  INFO("Testing MinmodTci in 2D");
  const auto minmod_type = SlopeLimiters::MinmodType::LambdaPi1;
  const double tvbm_constant = 0.0;
  const auto element = make_element<2>();
  const Mesh<2> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<2>(2.0);

  const auto test_activates =
      [&minmod_type, &tvbm_constant, &element, &mesh, &
       element_size ](const DataVector& local_input,
                      const std::array<double, 4>& neighbor_means,
                      const std::array<double, 2>& expected_slopes) noexcept {
    test_minmod_tci_activates(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_four_neighbors(neighbor_means),
        make_four_neighbors(make_array<4>(2.0)), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &tvbm_constant, &element, &mesh, &
       element_size ](const DataVector& local_input,
                      const std::array<double, 4>& neighbor_means,
                      const std::array<double, 2>& original_slopes) noexcept {
    test_minmod_tci_does_not_activate(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_four_neighbors(neighbor_means),
        make_four_neighbors(make_array<4>(2.0)), original_slopes);
  };

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    return DataVector{3.0 + x + 2.0 * y + 0.1 * x * y};
  }
  ();

  // Case with no activation
  test_does_not_activate(input, {{1.9, 4.2, -0.5, 5.6}}, {{1.0, 2.0}});

  // Limit because of xi-direction neighbors
  test_activates(input, {{2.2, 4.2, -0.5, 5.6}}, {{0.8, 2.0}});
  test_activates(input, {{1.9, 3.2, -0.5, 5.6}}, {{0.2, 2.0}});

  // Limit because of eta-direction neighbors
  test_activates(input, {{1.9, 4.2, 1.5, 5.6}}, {{1.0, 1.5}});
  test_activates(input, {{1.9, 4.2, -0.5, 2.9}}, {{1.0, 0.0}});

  // Limit for xi and eta directions
  test_activates(input, {{2.2, 4.2, 1.5, 5.6}}, {{0.8, 1.5}});
  test_activates(input, {{3.9, 4.2, -0.5, 2.9}}, {{0.0, 0.0}});
}

// In 2D, test that the dimension-by-dimension application of the TCI works as
// expected.
void test_minmod_tci_3d() noexcept {
  INFO("Testing MinmodTci in 3D");
  const auto minmod_type = SlopeLimiters::MinmodType::LambdaPi1;
  const double tvbm_constant = 0.0;
  const auto element = make_element<3>();
  const Mesh<3> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<3>(2.0);

  const auto test_activates =
      [&minmod_type, &tvbm_constant, &element, &mesh, &
       element_size ](const DataVector& local_input,
                      const std::array<double, 6>& neighbor_means,
                      const std::array<double, 3>& expected_slopes) noexcept {
    test_minmod_tci_activates(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_six_neighbors(neighbor_means),
        make_six_neighbors(make_array<6>(2.0)), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &tvbm_constant, &element, &mesh, &
       element_size ](const DataVector& local_input,
                      const std::array<double, 6>& neighbor_means,
                      const std::array<double, 3>& original_slopes) noexcept {
    test_minmod_tci_does_not_activate(
        minmod_type, tvbm_constant, local_input, element, mesh, element_size,
        make_six_neighbors(neighbor_means),
        make_six_neighbors(make_array<6>(2.0)), original_slopes);
  };

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    return DataVector{2.0 - 1.6 * x + 0.4 * y + 0.4 * z + 0.1 * x * y -
                      0.1 * x * z - 0.2 * y * z};
  }
  ();

  // Case with no activation
  test_does_not_activate(input, {{3.8, -0.1, 1.5, 2.7, 1.2, 2.5}},
                         {{-1.6, 0.4, 0.4}});

  // Limit because of xi-direction neighbors
  test_activates(input, {{3.4, -0.1, 1.5, 2.7, 1.2, 2.5}}, {{-1.4, 0.4, 0.4}});
  test_activates(input, {{3.8, 2.1, 1.5, 2.7, 1.2, 2.5}}, {{0.0, 0.4, 0.4}});

  // Limit because of eta-direction neighbors
  test_activates(input, {{3.8, -0.1, 1.9, 2.7, 1.2, 2.5}}, {{-1.6, 0.1, 0.4}});
  test_activates(input, {{3.8, -0.1, 1.5, 2.3, 1.2, 2.5}}, {{-1.6, 0.3, 0.4}});

  // Limit because of zeta-direction neighbors
  test_activates(input, {{3.8, -0.1, 1.5, 2.7, 2.2, 2.5}}, {{-1.6, 0.4, 0.0}});
  test_activates(input, {{3.8, -0.1, 1.5, 2.7, 1.2, 2.1}}, {{-1.6, 0.4, 0.1}});

  // Limit for xi, eta, and zeta directions
  test_activates(input, {{3.4, -0.1, 1.5, 2.3, 1.2, 2.1}}, {{-1.4, 0.3, 0.1}});
  test_activates(input, {{3.8, 2.1, 2.1, 2.7, 2.2, 2.5}}, {{0.0, 0.0, 0.0}});
}

struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Scalar"; }
};

template <size_t VolumeDim>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, VolumeDim>;
  static std::string name() noexcept { return "Vector"; }
};

void test_minmod_tci_several_tensors() noexcept {
  INFO("Testing MinmodTci action on several tensors");
  // Test that TCI returns true if just one component needs limiting, which
  // we do by limiting a scalar and vector in 3D
  const auto minmod_type = SlopeLimiters::MinmodType::LambdaPi1;
  const double tvbm_constant = 0.0;
  const auto element = make_element<3>();
  const size_t number_of_grid_points = 2;
  const Mesh<3> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<3>(1.0);

  const auto linear_data = [&mesh](const double mean, const double slope_x,
                                   const double slope_y,
                                   const double slope_z) noexcept {
    const auto coords = logical_coordinates(mesh);
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    return DataVector{mean + slope_x * x + slope_y * y + slope_z * z};
  };

  Scalar<DataVector> local_scalar{linear_data(1.8, 1.4, 0.1, -0.2)};
  tnsr::I<DataVector, 3> local_vector{
      {{linear_data(-0.4, -0.3, 0.3, 1.0), linear_data(0.02, 0.01, 0.01, 0.2),
        linear_data(2.3, 0.5, -0.3, -0.2)}}};

  struct TestPackagedData {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>, ::Tags::Mean<VectorTag<3>>>
        means;
    std::array<double, 3> element_size =
        make_array<3>(std::numeric_limits<double>::signaling_NaN());
  };
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>, TestPackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data{};
  TestPackagedData& lower_xi_neighbor =
      neighbor_data[std::make_pair(Direction<3>::lower_xi(), ElementId<3>(1))];
  TestPackagedData& upper_xi_neighbor =
      neighbor_data[std::make_pair(Direction<3>::upper_xi(), ElementId<3>(2))];
  TestPackagedData& lower_eta_neighbor =
      neighbor_data[std::make_pair(Direction<3>::lower_eta(), ElementId<3>(3))];
  TestPackagedData& upper_eta_neighbor =
      neighbor_data[std::make_pair(Direction<3>::upper_eta(), ElementId<3>(4))];
  TestPackagedData& lower_zeta_neighbor = neighbor_data[std::make_pair(
      Direction<3>::lower_zeta(), ElementId<3>(5))];
  TestPackagedData& upper_zeta_neighbor = neighbor_data[std::make_pair(
      Direction<3>::upper_zeta(), ElementId<3>(6))];
  lower_xi_neighbor.element_size = element_size;
  upper_xi_neighbor.element_size = element_size;
  lower_eta_neighbor.element_size = element_size;
  upper_eta_neighbor.element_size = element_size;
  lower_zeta_neighbor.element_size = element_size;
  upper_zeta_neighbor.element_size = element_size;

  // Case where neither tensor triggers limiting
  get(get<::Tags::Mean<ScalarTag>>(lower_xi_neighbor.means)) = 0.3;
  get(get<::Tags::Mean<ScalarTag>>(upper_xi_neighbor.means)) = 3.3;
  get(get<::Tags::Mean<ScalarTag>>(lower_eta_neighbor.means)) = 1.6;
  get(get<::Tags::Mean<ScalarTag>>(upper_eta_neighbor.means)) = 2.2;
  get(get<::Tags::Mean<ScalarTag>>(lower_zeta_neighbor.means)) = 2.1;
  get(get<::Tags::Mean<ScalarTag>>(upper_zeta_neighbor.means)) = 1.4;
  get<0>(get<::Tags::Mean<VectorTag<3>>>(lower_xi_neighbor.means)) = 0.1;
  get<0>(get<::Tags::Mean<VectorTag<3>>>(upper_xi_neighbor.means)) = -0.9;
  get<0>(get<::Tags::Mean<VectorTag<3>>>(lower_eta_neighbor.means)) = -1.1;
  get<0>(get<::Tags::Mean<VectorTag<3>>>(upper_eta_neighbor.means)) = 0.2;
  get<0>(get<::Tags::Mean<VectorTag<3>>>(lower_zeta_neighbor.means)) = -1.8;
  get<0>(get<::Tags::Mean<VectorTag<3>>>(upper_zeta_neighbor.means)) = 0.7;
  get<1>(get<::Tags::Mean<VectorTag<3>>>(lower_xi_neighbor.means)) = 0.0;
  get<1>(get<::Tags::Mean<VectorTag<3>>>(upper_xi_neighbor.means)) = 0.05;
  get<1>(get<::Tags::Mean<VectorTag<3>>>(lower_eta_neighbor.means)) = -0.1;
  get<1>(get<::Tags::Mean<VectorTag<3>>>(upper_eta_neighbor.means)) = 0.1;
  get<1>(get<::Tags::Mean<VectorTag<3>>>(lower_zeta_neighbor.means)) = -0.7;
  get<1>(get<::Tags::Mean<VectorTag<3>>>(upper_zeta_neighbor.means)) = 0.3;
  get<2>(get<::Tags::Mean<VectorTag<3>>>(lower_xi_neighbor.means)) = 1.4;
  get<2>(get<::Tags::Mean<VectorTag<3>>>(upper_xi_neighbor.means)) = 2.9;
  get<2>(get<::Tags::Mean<VectorTag<3>>>(lower_eta_neighbor.means)) = 2.7;
  get<2>(get<::Tags::Mean<VectorTag<3>>>(upper_eta_neighbor.means)) = 1.8;
  get<2>(get<::Tags::Mean<VectorTag<3>>>(lower_zeta_neighbor.means)) = 3.1;
  get<2>(get<::Tags::Mean<VectorTag<3>>>(upper_zeta_neighbor.means)) = 0.5;
  bool activation = SlopeLimiters::Minmod_detail::troubled_cell_indicator<
      3, TestPackagedData, ScalarTag, VectorTag<3>>(
      local_scalar, local_vector, neighbor_data, minmod_type, tvbm_constant,
      element, mesh, element_size);
  CHECK_FALSE(activation);

  // Case where the scalar triggers limiting
  get(get<::Tags::Mean<ScalarTag>>(upper_xi_neighbor.means)) = 2.0;
  activation = SlopeLimiters::Minmod_detail::troubled_cell_indicator<
      3, TestPackagedData, ScalarTag, VectorTag<3>>(
      local_scalar, local_vector, neighbor_data, minmod_type, tvbm_constant,
      element, mesh, element_size);
  CHECK(activation);

  // Case where the vector x-component triggers limiting
  get(get<::Tags::Mean<ScalarTag>>(upper_xi_neighbor.means)) = 3.3;
  get<0>(get<::Tags::Mean<VectorTag<3>>>(lower_zeta_neighbor.means)) = -0.1;
  activation = SlopeLimiters::Minmod_detail::troubled_cell_indicator<
      3, TestPackagedData, ScalarTag, VectorTag<3>>(
      local_scalar, local_vector, neighbor_data, minmod_type, tvbm_constant,
      element, mesh, element_size);
  CHECK(activation);

  // Case where the vector y-component triggers limiting
  get<0>(get<::Tags::Mean<VectorTag<3>>>(lower_zeta_neighbor.means)) = -1.8;
  get<1>(get<::Tags::Mean<VectorTag<3>>>(upper_eta_neighbor.means)) = -0.2;
  activation = SlopeLimiters::Minmod_detail::troubled_cell_indicator<
      3, TestPackagedData, ScalarTag, VectorTag<3>>(
      local_scalar, local_vector, neighbor_data, minmod_type, tvbm_constant,
      element, mesh, element_size);
  CHECK(activation);

  // Case where the vector z-component triggers limiting
  get<1>(get<::Tags::Mean<VectorTag<3>>>(upper_eta_neighbor.means)) = 0.1;
  get<2>(get<::Tags::Mean<VectorTag<3>>>(lower_xi_neighbor.means)) = 1.9;
  activation = SlopeLimiters::Minmod_detail::troubled_cell_indicator<
      3, TestPackagedData, ScalarTag, VectorTag<3>>(
      local_scalar, local_vector, neighbor_data, minmod_type, tvbm_constant,
      element, mesh, element_size);
  CHECK(activation);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.MinmodTci",
                  "[SlopeLimiters][Unit]") {
  test_minmod_tci_1d();
  test_minmod_tci_2d();
  test_minmod_tci_3d();

  test_minmod_tci_several_tensors();
}
