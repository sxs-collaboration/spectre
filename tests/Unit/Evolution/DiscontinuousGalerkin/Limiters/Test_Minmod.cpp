// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.tpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
// IWYU pragma: no_forward_declare Limiters::Minmod
// IWYU pragma: no_forward_declare Tensor

namespace {

struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Scalar"; }
};

template <size_t VolumeDim>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, VolumeDim>;
  static std::string name() noexcept { return "Vector"; }
};

void test_minmod_option_parsing() noexcept {
  INFO("Test Minmod option parsing");
  const auto lambda_pi1 =
      TestHelpers::test_creation<Limiters::Minmod<1, tmpl::list<ScalarTag>>>(
          "Type: LambdaPi1\n"
          "TvbConstant: 0.0");
  const auto lambda_pi1_tvb =
      TestHelpers::test_creation<Limiters::Minmod<1, tmpl::list<ScalarTag>>>(
          "Type: LambdaPi1\n"
          "TvbConstant: 1.0");
  const auto lambda_pi1_disabled =
      TestHelpers::test_creation<Limiters::Minmod<1, tmpl::list<ScalarTag>>>(
          "Type: LambdaPi1\n"
          "TvbConstant: 0.0\n"
          "DisableForDebugging: True");
  const auto muscl =
      TestHelpers::test_creation<Limiters::Minmod<1, tmpl::list<ScalarTag>>>(
          "Type: Muscl\n"
          "TvbConstant: 0.0");

  // Test operators == and !=
  CHECK(lambda_pi1 == lambda_pi1);
  CHECK(lambda_pi1 != lambda_pi1_tvb);
  CHECK(lambda_pi1 != lambda_pi1_disabled);
  CHECK(lambda_pi1 != muscl);

  const auto lambda_pin_2d =
      TestHelpers::test_creation<Limiters::Minmod<2, tmpl::list<ScalarTag>>>(
          "Type: LambdaPiN\n"
          "TvbConstant: 0.0");
  const auto lambda_pin_3d = TestHelpers::test_creation<
      Limiters::Minmod<3, tmpl::list<ScalarTag, VectorTag<3>>>>(
      "Type: LambdaPiN\n"
      "TvbConstant: 10.0");

  // Test that creation from options gives correct object
  const Limiters::Minmod<1, tmpl::list<ScalarTag>> expected_lambda_pi1(
      Limiters::MinmodType::LambdaPi1, 0.0);
  const Limiters::Minmod<1, tmpl::list<ScalarTag>> expected_lambda_pi1_tvb(
      Limiters::MinmodType::LambdaPi1, 1.0);
  const Limiters::Minmod<1, tmpl::list<ScalarTag>> expected_lambda_pi1_disabled(
      Limiters::MinmodType::LambdaPi1, 0.0, true);
  const Limiters::Minmod<1, tmpl::list<ScalarTag>> expected_muscl(
      Limiters::MinmodType::Muscl, 0.0);
  const Limiters::Minmod<2, tmpl::list<ScalarTag>> expected_lambda_pin_2d(
      Limiters::MinmodType::LambdaPiN, 0.0);
  const Limiters::Minmod<3, tmpl::list<ScalarTag, VectorTag<3>>>
      expected_lambda_pin_3d(Limiters::MinmodType::LambdaPiN, 10.0);

  CHECK(lambda_pi1 == expected_lambda_pi1);
  CHECK(lambda_pi1_tvb == expected_lambda_pi1_tvb);
  CHECK(lambda_pi1_disabled == expected_lambda_pi1_disabled);
  CHECK(muscl == expected_muscl);
  CHECK(lambda_pin_2d == expected_lambda_pin_2d);
  CHECK(lambda_pin_3d == expected_lambda_pin_3d);
}

void test_minmod_serialization() noexcept {
  INFO("Test Minmod serialization");
  const Limiters::Minmod<1, tmpl::list<ScalarTag>> minmod(
      Limiters::MinmodType::LambdaPi1, 1.0);
  test_serialization(minmod);
}

template <size_t VolumeDim>
void test_package_data_work(
    const Mesh<VolumeDim>& mesh,
    const OrientationMap<VolumeDim>& orientation_map) noexcept {
  const DataVector used_for_size(mesh.number_of_grid_points());
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);

  const auto input_scalar = make_with_random_values<ScalarTag::type>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto input_vector =
      make_with_random_values<typename VectorTag<VolumeDim>::type>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto element_size =
      make_with_random_values<std::array<double, VolumeDim>>(
          make_not_null(&generator), make_not_null(&dist), 0.0);

  using TagList = tmpl::list<ScalarTag, VectorTag<VolumeDim>>;
  const double tvb_constant = 0.0;
  const Limiters::Minmod<VolumeDim, TagList> minmod(
      Limiters::MinmodType::LambdaPiN, tvb_constant);
  typename Limiters::Minmod<VolumeDim, TagList>::PackagedData packaged_data{};

  minmod.package_data(make_not_null(&packaged_data), input_scalar, input_vector,
                      mesh, element_size, orientation_map);

  CHECK(packaged_data.element_size ==
        orientation_map.permute_from_neighbor(element_size));

  // Means are just numbers and don't care about orientations
  CHECK(get(get<::Tags::Mean<ScalarTag>>(packaged_data.means)) ==
        mean_value(get(input_scalar), mesh));
  for (size_t i = 0; i < VolumeDim; ++i) {
    CHECK(get<::Tags::Mean<VectorTag<VolumeDim>>>(packaged_data.means).get(i) ==
          mean_value(input_vector.get(i), mesh));
  }
}

void test_package_data_1d() noexcept {
  INFO("Test Minmod package_data in 1D");
  const Mesh<1> mesh(4, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  const OrientationMap<1> orientation_aligned{};
  test_package_data_work(mesh, orientation_aligned);

  const OrientationMap<1> orientation_flipped(
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
  test_package_data_work(mesh, orientation_flipped);
}

void test_package_data_2d() noexcept {
  INFO("Test WENO package_data in 2D");
  const Mesh<2> mesh({{4, 6}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  const OrientationMap<2> orientation_aligned{};
  test_package_data_work(mesh, orientation_aligned);

  const OrientationMap<2> orientation_rotated(std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}});
  test_package_data_work(mesh, orientation_rotated);
}

void test_package_data_3d() noexcept {
  INFO("Test WENO package_data in 3D");
  const Mesh<3> mesh({{4, 6, 3}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  const OrientationMap<3> orientation_aligned{};
  test_package_data_work(mesh, orientation_aligned);

  const OrientationMap<3> orientation_rotated(std::array<Direction<3>, 3>{
      {Direction<3>::lower_zeta(), Direction<3>::upper_xi(),
       Direction<3>::lower_eta()}});
  test_package_data_work(mesh, orientation_rotated);
}

// Helper function to wrap the allocation of the optimization buffers for the
// troubled cell indicator function.
template <size_t VolumeDim>
bool wrap_minmod_limited_slopes(
    const gsl::not_null<double*> u_mean,
    const gsl::not_null<std::array<double, VolumeDim>*> u_limited_slopes,
    const Limiters::MinmodType& minmod_type, const double tvb_constant,
    const DataVector& u, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes) noexcept {
  DataVector u_lin_buffer(mesh.number_of_grid_points());
  Limiters::Minmod_detail::BufferWrapper<VolumeDim> buffer(mesh);
  return Limiters::Minmod_detail::minmod_limited_slopes(
      make_not_null(&u_lin_buffer), make_not_null(&buffer), u_mean,
      u_limited_slopes, minmod_type, tvb_constant, u, mesh, element,
      element_size, effective_neighbor_means, effective_neighbor_sizes);
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
void test_minmod_activates(
    const Limiters::MinmodType& minmod_type, const double tvb_constant,
    const DataVector& input, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes,
    const std::array<double, VolumeDim>& expected_slopes) noexcept {
  double u_mean{};
  std::array<double, VolumeDim> u_limited_slopes{};
  const bool reduce_slopes = wrap_minmod_limited_slopes(
      make_not_null(&u_mean), make_not_null(&u_limited_slopes), minmod_type,
      tvb_constant, input, mesh, element, element_size,
      effective_neighbor_means, effective_neighbor_sizes);
  CHECK(reduce_slopes);
  CHECK(u_mean == approx(mean_value(input, mesh)));
  CHECK_ITERABLE_APPROX(u_limited_slopes, expected_slopes);
}

// Test that TCI does not detect a cell that is not troubled.
template <size_t VolumeDim>
void test_minmod_does_not_activate(
    const Limiters::MinmodType& minmod_type, const double tvb_constant,
    const DataVector& input, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes,
    const std::array<double, VolumeDim>& original_slopes) noexcept {
  double u_mean{};
  std::array<double, VolumeDim> u_limited_slopes{};
  const bool reduce_slopes = wrap_minmod_limited_slopes(
      make_not_null(&u_mean), make_not_null(&u_limited_slopes), minmod_type,
      tvb_constant, input, mesh, element, element_size,
      effective_neighbor_means, effective_neighbor_sizes);
  CHECK_FALSE(reduce_slopes);
  if (minmod_type != Limiters::MinmodType::LambdaPiN) {
    CHECK(u_mean == approx(mean_value(input, mesh)));
    CHECK_ITERABLE_APPROX(u_limited_slopes, original_slopes);
  }
}

void test_minmod_slopes_on_linear_function(
    const size_t number_of_grid_points,
    const Limiters::MinmodType& minmod_type) noexcept {
  INFO("Testing linear function...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const double tvb_constant = 0.0;
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto element_size = make_array<1>(2.0);

  const auto test_activates = [&minmod_type, &tvb_constant, &mesh, &element,
                               &element_size](
                                  const DataVector& local_input,
                                  const double left, const double right,
                                  const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_activates(minmod_type, tvb_constant, local_input, mesh, element,
                          element_size, make_two_neighbors(left, right),
                          make_two_neighbors(2.0, 2.0), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &tvb_constant, &mesh, &element, &element_size](
          const DataVector& local_input, const double left, const double right,
          const double original_slope) noexcept {
        const auto original_slopes = make_array<1>(original_slope);
        test_minmod_does_not_activate(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
            make_two_neighbors(left, right), make_two_neighbors(2.0, 2.0),
            original_slopes);
      };

  // With a MUSCL limiter, the largest allowed slope is half as big as for a
  // LambdaPi1 or LambdaPiN limiter. We can re-use the same test cases by
  // correspondingly scaling the slope:
  const double muscl_slope_factor =
      (minmod_type == Limiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  // Test a positive-slope function
  {
    const auto input = [&muscl_slope_factor, &mesh]() noexcept {
      const auto coords = logical_coordinates(mesh);
      return DataVector{3.6 + 1.2 * muscl_slope_factor * get<0>(coords)};
    }();

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
    const auto input = [&muscl_slope_factor, &mesh]() noexcept {
      const auto coords = logical_coordinates(mesh);
      return DataVector{-0.4 - 0.8 * muscl_slope_factor * get<0>(coords)};
    }();

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

void test_minmod_slopes_on_quadratic_function(
    const size_t number_of_grid_points,
    const Limiters::MinmodType& minmod_type) noexcept {
  INFO("Testing quadratic function...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const double tvb_constant = 0.0;
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto element_size = make_array<1>(2.0);

  const auto test_activates = [&minmod_type, &tvb_constant, &mesh, &element,
                               &element_size](
                                  const DataVector& local_input,
                                  const double left, const double right,
                                  const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_activates(minmod_type, tvb_constant, local_input, mesh, element,
                          element_size, make_two_neighbors(left, right),
                          make_two_neighbors(2.0, 2.0), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &tvb_constant, &mesh, &element, &element_size](
          const DataVector& local_input, const double left, const double right,
          const double original_slope) noexcept {
        const auto original_slopes = make_array<1>(original_slope);
        test_minmod_does_not_activate(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
            make_two_neighbors(left, right), make_two_neighbors(2.0, 2.0),
            original_slopes);
      };

  const double muscl_slope_factor =
      (minmod_type == Limiters::MinmodType::Muscl) ? 0.5 : 1.0;

  const auto input = [&muscl_slope_factor, &mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    const auto& x = get<0>(coords);
    // For easier testing, center the quadratic term on the grid: otherwise this
    // term will affect the average slope on the element.
    return DataVector{13.0 + 4.0 * muscl_slope_factor * x + 2.5 * square(x)};
  }();
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

void test_minmod_slopes_with_tvb_correction(
    const size_t number_of_grid_points,
    const Limiters::MinmodType& minmod_type) noexcept {
  INFO("Testing TVB correction...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto element_size = make_array<1>(2.0);

  const auto test_activates = [&minmod_type, &mesh, &element, &element_size](
                                  const double tvb_constant,
                                  const DataVector& local_input,
                                  const double left, const double right,
                                  const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_activates(minmod_type, tvb_constant, local_input, mesh, element,
                          element_size, make_two_neighbors(left, right),
                          make_two_neighbors(2.0, 2.0), expected_slopes);
  };
  const auto test_does_not_activate =
      [&minmod_type, &mesh, &element, &element_size](
          const double tvb_constant, const DataVector& local_input,
          const double left, const double right,
          const double original_slope) noexcept {
        const auto original_slopes = make_array<1>(original_slope);
        test_minmod_does_not_activate(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
            make_two_neighbors(left, right), make_two_neighbors(2.0, 2.0),
            original_slopes);
      };

  // Slopes will be compared to m * h^2, where here h = 2
  const double tvb_m0 = 0.0;
  const double tvb_m1 = 1.0;
  const double tvb_m2 = 2.0;

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{21.6 + 7.2 * get<0>(coords)};
  }();

  // The TVB constant sets a threshold slope, below which the solution will not
  // be limited. We test this by increasing the TVB constant until the limiter
  // stops activating / stops changing the solution.
  const double left = (minmod_type == Limiters::MinmodType::Muscl) ? 8.4 : 15.0;
  const double right =
      (minmod_type == Limiters::MinmodType::Muscl) ? 34.8 : 30.0;
  test_activates(tvb_m0, input, left, right, 6.6);
  test_activates(tvb_m1, input, left, right, 6.6);
  test_does_not_activate(tvb_m2, input, left, right, 7.2);
}

// Here we test the coupling of the LambdaPiN troubled cell detector with the
// TVB constant value.
void test_lambda_pin_troubled_cell_tvb_correction(
    const size_t number_of_grid_points) noexcept {
  INFO("Testing LambdaPiN-TVB correction...");
  CAPTURE(number_of_grid_points);
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(2.0);

  const auto test_activates = [&mesh, &element, &element_size](
                                  const double tvb_constant,
                                  const DataVector& local_input,
                                  const double left, const double right,
                                  const double expected_slope) noexcept {
    const auto expected_slopes = make_array<1>(expected_slope);
    test_minmod_activates(Limiters::MinmodType::LambdaPiN, tvb_constant,
                          local_input, mesh, element, element_size,
                          make_two_neighbors(left, right),
                          make_two_neighbors(2.0, 2.0), expected_slopes);
  };
  const auto test_does_not_activate =
      [&mesh, &element, &element_size](
          const double tvb_constant, const DataVector& local_input,
          const double left, const double right) noexcept {
        // Because in this test the limiter is LambdaPiN, no slopes are
        // returned, and no comparison is made. So set these to NaN
        const auto original_slopes =
            make_array<1>(std::numeric_limits<double>::signaling_NaN());
        test_minmod_does_not_activate(
            Limiters::MinmodType::LambdaPiN, tvb_constant, local_input, mesh,
            element, element_size, make_two_neighbors(left, right),
            make_two_neighbors(2.0, 2.0), original_slopes);
      };

  const double m0 = 0.0;
  const double m1 = 1.0;
  const double m2 = 2.0;

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{10.0 * step_function(get<0>(coords))};
  }();

  // Establish baseline m = 0 case; LambdaPiN normally avoids limiting when
  // max(edge - mean) <= min(neighbor - mean)
  test_does_not_activate(m0, input, 0.0, 10.0);
  test_does_not_activate(m0, input, -0.3, 10.2);
  // but does limit if max(edge - mean) > min(neighbor - mean)
  test_activates(m0, input, 0.02, 10.0, 4.98);
  test_activates(m0, input, 0.0, 9.99, 4.99);

  // However, with a non-zero TVB constant, LambdaPiN should additionally avoid
  // limiting when max(edge - mean) < TVB correction.
  // We test first a case where the TVB correction is too small to affect
  // the limiter action,
  test_does_not_activate(m1, input, 0.0, 10.0);
  test_does_not_activate(m1, input, -0.3, 10.2);
  test_activates(m1, input, 0.02, 10.0, 4.98);
  test_activates(m1, input, 0.0, 9.99, 4.99);

  // And a case where the TVB correction enables LambdaPiN to avoid limiting
  // (Note that the slope here is still too large to avoid limiting through
  // the normal TVB tolerance.)
  test_does_not_activate(m2, input, 0.0, 10.0);
  test_does_not_activate(m2, input, -0.3, 10.2);
  test_does_not_activate(m2, input, 0.02, 10.0);
  test_does_not_activate(m2, input, 0.0, 9.99);
}

void test_minmod_slopes_at_boundary(
    const size_t number_of_grid_points,
    const Limiters::MinmodType& minmod_type) noexcept {
  INFO("Testing limiter at boundary...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const double tvb_constant = 0.0;
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element_size = make_array<1>(2.0);

  const double muscl_slope_factor =
      (minmod_type == Limiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const auto input = [&muscl_slope_factor, &mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{1.2 * muscl_slope_factor * get<0>(coords)};
  }();

  // Test with element that has external lower-xi boundary
  const auto element_at_lower_xi_boundary =
      TestHelpers::Limiters::make_element<1>({{Direction<1>::lower_xi()}});
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_minmod_activates(
        minmod_type, tvb_constant, input, mesh, element_at_lower_xi_boundary,
        element_size, {{std::make_pair(Direction<1>::upper_xi(), neighbor)}},
        {{std::make_pair(Direction<1>::upper_xi(), element_size[0])}},
        make_array<1>(0.0));
  }

  // Test with element that has external upper-xi boundary
  const auto element_at_upper_xi_boundary =
      TestHelpers::Limiters::make_element<1>({{Direction<1>::upper_xi()}});
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_minmod_activates(
        minmod_type, tvb_constant, input, mesh, element_at_upper_xi_boundary,
        element_size, {{std::make_pair(Direction<1>::lower_xi(), neighbor)}},
        {{std::make_pair(Direction<1>::lower_xi(), element_size[0])}},
        make_array<1>(0.0));
  }
}

void test_minmod_slopes_with_different_size_neighbor(
    const size_t number_of_grid_points,
    const Limiters::MinmodType& minmod_type) noexcept {
  INFO("Testing limiter with neighboring elements of different size...");
  CAPTURE(number_of_grid_points);
  CAPTURE(get_output(minmod_type));
  const double tvb_constant = 0.0;
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<1>();
  const double dx = 1.0;
  const auto element_size = make_array<1>(dx);

  const auto test_activates =
      [&minmod_type, &tvb_constant, &mesh, &element, &element_size](
          const DataVector& local_input, const double left, const double right,
          const double left_size, const double right_size,
          const double expected_slope) noexcept {
        const auto expected_slopes = make_array<1>(expected_slope);
        test_minmod_activates(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
            make_two_neighbors(left, right),
            make_two_neighbors(left_size, right_size), expected_slopes);
      };
  const auto test_does_not_activate =
      [&minmod_type, &tvb_constant, &mesh, &element, &element_size](
          const DataVector& local_input, const double left, const double right,
          const double left_size, const double right_size,
          const double original_slope) noexcept {
        const auto original_slopes = make_array<1>(original_slope);
        test_minmod_does_not_activate(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
            make_two_neighbors(left, right),
            make_two_neighbors(left_size, right_size), original_slopes);
      };

  const double muscl_slope_factor =
      (minmod_type == Limiters::MinmodType::Muscl) ? 0.5 : 1.0;
  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  const auto input = [&muscl_slope_factor, &mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{2.0 + 1.2 * muscl_slope_factor * get<0>(coords)};
  }();

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

// In 1D, test combinations of MinmodType, TVB constant, polynomial order, etc.
// Check that each combination reduces the slopes as expected.
void test_minmod_limited_slopes_1d() noexcept {
  INFO("Testing Minmod minmod_limited_slopes in 1D");
  for (const auto& minmod_type :
       {Limiters::MinmodType::LambdaPi1, Limiters::MinmodType::LambdaPiN,
        Limiters::MinmodType::Muscl}) {
    for (const auto num_grid_points : std::array<size_t, 2>{{2, 4}}) {
      test_minmod_slopes_on_linear_function(num_grid_points, minmod_type);
      test_minmod_slopes_with_tvb_correction(num_grid_points, minmod_type);
      test_minmod_slopes_at_boundary(num_grid_points, minmod_type);
      test_minmod_slopes_with_different_size_neighbor(num_grid_points,
                                                      minmod_type);
    }
    // This test only makes sense with more than 2 grid points
    test_minmod_slopes_on_quadratic_function(3, minmod_type);
    test_minmod_slopes_on_quadratic_function(4, minmod_type);
  }
  // This test only makes sense with LambdaPiN and more than 2 grid points
  test_lambda_pin_troubled_cell_tvb_correction(4);
}

// In 2D, test that the slopes are correctly reduced dimension-by-dimension.
void test_minmod_limited_slopes_2d() noexcept {
  INFO("Testing Minmod minmod_limited_slopes in 2D");
  const auto minmod_type = Limiters::MinmodType::LambdaPi1;
  const double tvb_constant = 0.0;
  const Mesh<2> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<2>();
  const auto element_size = make_array<2>(2.0);

  const auto test_activates =
      [&minmod_type, &tvb_constant, &mesh, &element, &element_size](
          const DataVector& local_input,
          const std::array<double, 4>& neighbor_means,
          const std::array<double, 2>& expected_slopes) noexcept {
        test_minmod_activates(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
            make_four_neighbors(neighbor_means),
            make_four_neighbors(make_array<4>(2.0)), expected_slopes);
      };
  const auto test_does_not_activate =
      [&minmod_type, &tvb_constant, &mesh, &element, &element_size](
          const DataVector& local_input,
          const std::array<double, 4>& neighbor_means,
          const std::array<double, 2>& original_slopes) noexcept {
        test_minmod_does_not_activate(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
            make_four_neighbors(neighbor_means),
            make_four_neighbors(make_array<4>(2.0)), original_slopes);
      };

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    return DataVector{3.0 + x + 2.0 * y + 0.1 * x * y};
  }();

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

// In 3D, test that the slopes are correctly reduced dimension-by-dimension.
void test_minmod_limited_slopes_3d() noexcept {
  INFO("Testing Minmod minmod_limited_slopes in 3D");
  const auto minmod_type = Limiters::MinmodType::LambdaPi1;
  const double tvb_constant = 0.0;
  const Mesh<3> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<3>();
  const auto element_size = make_array<3>(2.0);

  const auto test_activates =
      [&minmod_type, &tvb_constant, &mesh, &element, &element_size](
          const DataVector& local_input,
          const std::array<double, 6>& neighbor_means,
          const std::array<double, 3>& expected_slopes) noexcept {
        test_minmod_activates(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
            make_six_neighbors(neighbor_means),
            make_six_neighbors(make_array<6>(2.0)), expected_slopes);
      };
  const auto test_does_not_activate =
      [&minmod_type, &tvb_constant, &mesh, &element, &element_size](
          const DataVector& local_input,
          const std::array<double, 6>& neighbor_means,
          const std::array<double, 3>& original_slopes) noexcept {
        test_minmod_does_not_activate(
            minmod_type, tvb_constant, local_input, mesh, element, element_size,
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
  }();

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

// Helper function for testing Minmod::op()
template <size_t VolumeDim>
void test_limiter_work(
    const Scalar<DataVector>& input_scalar,
    const tnsr::I<DataVector, VolumeDim>& input_vector,
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename Limiters::Minmod<
            VolumeDim,
            tmpl::list<ScalarTag, VectorTag<VolumeDim>>>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const std::array<double, VolumeDim>& target_scalar_slope,
    const std::array<std::array<double, VolumeDim>, VolumeDim>&
        target_vector_slope) noexcept {
  auto scalar_to_limit = input_scalar;
  auto vector_to_limit = input_vector;

  // Minmod should preserve the mean, so expected = initial
  const double expected_scalar_mean = mean_value(get(scalar_to_limit), mesh);
  const auto expected_vector_means = [&vector_to_limit, &mesh]() noexcept {
    std::array<double, VolumeDim> means{};
    for (size_t d = 0; d < VolumeDim; ++d) {
      gsl::at(means, d) = mean_value(vector_to_limit.get(d), mesh);
    }
    return means;
  }();

  const auto element = TestHelpers::Limiters::make_element<VolumeDim>();
  const double tvb_constant = 0.0;
  const Limiters::Minmod<VolumeDim, tmpl::list<ScalarTag, VectorTag<VolumeDim>>>
      minmod(Limiters::MinmodType::LambdaPi1, tvb_constant);
  const bool limiter_activated =
      minmod(make_not_null(&scalar_to_limit), make_not_null(&vector_to_limit),
             mesh, element, logical_coords, element_size, neighbor_data);

  CHECK(limiter_activated);

  CHECK(mean_value(get(scalar_to_limit), mesh) == approx(expected_scalar_mean));
  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK(mean_value(vector_to_limit.get(d), mesh) ==
          approx(gsl::at(expected_vector_means, d)));
  }

  const auto expected_limiter_output =
      [&logical_coords, &mesh](
          const DataVector& input,
          const std::array<double, VolumeDim> expected_slope) noexcept {
        auto result =
            make_with_value<DataVector>(input, mean_value(input, mesh));
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

void test_minmod_limiter_1d() noexcept {
  INFO("Test Minmod limiter in 1D");
  // This test checks that Minmod limits different tensor components
  // independently
  //
  // We fill each local tensor component with the same volume data
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(0.5);
  const auto true_slope = std::array<double, 1>{{2.0}};
  const auto func =
      [&true_slope](
          const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
        const auto& x = get<0>(coords);
        return 1.0 + true_slope[0] * x + 3.3 * square(x);
      };
  const auto data = DataVector{func(logical_coords)};
  const double mean = mean_value(data, mesh);
  const auto input_scalar = ScalarTag::type{data};
  const auto input_vector = VectorTag<1>::type{data};

  // We fill the neighbor mean data with different values for each tensor
  // component, so that each component is limited in a different way
  std::unordered_map<
      std::pair<Direction<1>, ElementId<1>>,
      Limiters::Minmod<1, tmpl::list<ScalarTag, VectorTag<1>>>::PackagedData,
      boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<1>, ElementId<1>>, 2> dir_keys = {
      {{Direction<1>::lower_xi(), ElementId<1>(1)},
       {Direction<1>::upper_xi(), ElementId<1>(2)}}};
  neighbor_data[dir_keys[0]].element_size = element_size;
  neighbor_data[dir_keys[1]].element_size = element_size;

  // The scalar we treat as a shock: we want the slope to be reduced
  const auto target_scalar_slope = std::array<double, 1>{{1.2}};
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[0]].means) =
      Scalar<double>(mean - target_scalar_slope[0]);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[1]].means) =
      Scalar<double>(mean + target_scalar_slope[0]);

  // The vector x-component we treat as a smooth function: no limiter action
  const auto target_vector_slope =
      std::array<std::array<double, 1>, 1>{{true_slope}};
  get<Tags::Mean<VectorTag<1>>>(neighbor_data[dir_keys[0]].means) =
      tnsr::I<double, 1>(mean - 2.0 * true_slope[0]);
  get<Tags::Mean<VectorTag<1>>>(neighbor_data[dir_keys[1]].means) =
      tnsr::I<double, 1>(mean + 2.0 * true_slope[0]);

  test_limiter_work(input_scalar, input_vector, mesh, logical_coords,
                    element_size, neighbor_data, target_scalar_slope,
                    target_vector_slope);
}

void test_minmod_limiter_2d() noexcept {
  INFO("Test Minmod limiter in 2D");
  // This test checks that Minmod limits...
  // - different tensor components independently
  // - different dimensions independently
  //
  // We fill each local tensor component with the same volume data
  const auto mesh =
      Mesh<2>(std::array<size_t, 2>{{3, 3}}, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array(0.5, 1.0);
  const auto true_slope = std::array<double, 2>{{2.0, -3.0}};
  const auto& func =
      [&true_slope](
          const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
        const auto& x = get<0>(coords);
        const auto& y = get<1>(coords);
        return 1.0 + true_slope[0] * x + 3.3 * square(x) + true_slope[1] * y +
               square(y);
      };
  const auto data = DataVector{func(logical_coords)};
  const double mean = mean_value(data, mesh);
  const auto input_scalar = ScalarTag::type{data};
  const auto input_vector = VectorTag<2>::type{data};

  // We fill the neighbor mean data with different values for each tensor
  // component, so that each component is limited in a different way
  std::unordered_map<
      std::pair<Direction<2>, ElementId<2>>,
      Limiters::Minmod<2, tmpl::list<ScalarTag, VectorTag<2>>>::PackagedData,
      boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  const std::array<std::pair<Direction<2>, ElementId<2>>, 4> dir_keys = {
      {{Direction<2>::lower_xi(), ElementId<2>(1)},
       {Direction<2>::upper_xi(), ElementId<2>(2)},
       {Direction<2>::lower_eta(), ElementId<2>(3)},
       {Direction<2>::upper_eta(), ElementId<2>(4)}}};
  neighbor_data[dir_keys[0]].element_size = element_size;
  neighbor_data[dir_keys[1]].element_size = element_size;
  neighbor_data[dir_keys[2]].element_size = element_size;
  neighbor_data[dir_keys[3]].element_size = element_size;

  // The scalar we treat as a 3D shock: we want each slope to be reduced
  const auto target_scalar_slope = std::array<double, 2>{{1.2, -2.2}};
  const auto neighbor_scalar_func = [&mean, &target_scalar_slope](
                                        const size_t dim, const int sign) {
    return Scalar<double>(mean + sign * gsl::at(target_scalar_slope, dim));
  };
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[0]].means) =
      neighbor_scalar_func(0, -1);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[1]].means) =
      neighbor_scalar_func(0, 1);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[2]].means) =
      neighbor_scalar_func(1, -1);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[3]].means) =
      neighbor_scalar_func(1, 1);

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
  get<Tags::Mean<VectorTag<2>>>(neighbor_data[dir_keys[0]].means) =
      neighbor_vector_func(0, -1);
  get<Tags::Mean<VectorTag<2>>>(neighbor_data[dir_keys[1]].means) =
      neighbor_vector_func(0, 1);
  get<Tags::Mean<VectorTag<2>>>(neighbor_data[dir_keys[2]].means) =
      neighbor_vector_func(1, -1);
  get<Tags::Mean<VectorTag<2>>>(neighbor_data[dir_keys[3]].means) =
      neighbor_vector_func(1, 1);

  test_limiter_work(input_scalar, input_vector, mesh, logical_coords,
                    element_size, neighbor_data, target_scalar_slope,
                    target_vector_slope);
}

void test_minmod_limiter_3d() noexcept {
  INFO("Test Minmod limiter in 3D");
  // This test checks that Minmod limits...
  // - different tensor components independently
  // - different dimensions independently
  //
  // We fill each local tensor component with the same volume data
  const auto mesh =
      Mesh<3>(std::array<size_t, 3>{{3, 3, 4}}, Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array(0.5, 1.0, 0.8);
  const auto true_slope = std::array<double, 3>{{2.0, -3.0, 1.0}};
  const auto func =
      [&true_slope](
          const tnsr::I<DataVector, 3, Frame::Logical>& coords) noexcept {
        const auto& x = get<0>(coords);
        const auto& y = get<1>(coords);
        const auto& z = get<2>(coords);
        return 1.0 + true_slope[0] * x + 3.3 * square(x) + true_slope[1] * y +
               square(y) + true_slope[2] * z - square(z);
      };
  const auto data = DataVector{func(logical_coords)};
  const double mean = mean_value(data, mesh);
  const auto input_scalar = ScalarTag::type{data};
  const auto input_vector = VectorTag<3>::type{data};

  // We fill the neighbor mean data with different values for each tensor
  // component, so that each component is limited in a different way
  std::unordered_map<
      std::pair<Direction<3>, ElementId<3>>,
      Limiters::Minmod<3, tmpl::list<ScalarTag, VectorTag<3>>>::PackagedData,
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
    neighbor_data[id_pair].element_size = element_size;
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
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[0]].means) =
      neighbor_scalar_func(0, -1);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[1]].means) =
      neighbor_scalar_func(0, 1);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[2]].means) =
      neighbor_scalar_func(1, -1);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[3]].means) =
      neighbor_scalar_func(1, 1);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[4]].means) =
      neighbor_scalar_func(2, -1);
  get<Tags::Mean<ScalarTag>>(neighbor_data[dir_keys[5]].means) =
      neighbor_scalar_func(2, 1);

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
  get<Tags::Mean<VectorTag<3>>>(neighbor_data[dir_keys[0]].means) =
      neighbor_vector_func(0, -1);
  get<Tags::Mean<VectorTag<3>>>(neighbor_data[dir_keys[1]].means) =
      neighbor_vector_func(0, 1);
  get<Tags::Mean<VectorTag<3>>>(neighbor_data[dir_keys[2]].means) =
      neighbor_vector_func(1, -1);
  get<Tags::Mean<VectorTag<3>>>(neighbor_data[dir_keys[3]].means) =
      neighbor_vector_func(1, 1);
  get<Tags::Mean<VectorTag<3>>>(neighbor_data[dir_keys[4]].means) =
      neighbor_vector_func(2, -1);
  get<Tags::Mean<VectorTag<3>>>(neighbor_data[dir_keys[5]].means) =
      neighbor_vector_func(2, 1);

  test_limiter_work(input_scalar, input_vector, mesh, logical_coords,
                    element_size, neighbor_data, target_scalar_slope,
                    target_vector_slope);
}

// Test that the limiter activates in the x-direction only. Domain quantities
// and input Scalar may be of higher dimension VolumeDim.
template <size_t VolumeDim>
void test_limiter_activates_work(
    const Limiters::Minmod<VolumeDim, tmpl::list<ScalarTag>>& minmod,
    const ScalarTag::type& input, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename Limiters::Minmod<VolumeDim,
                                  tmpl::list<ScalarTag>>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const double expected_slope) noexcept {
  auto input_to_limit = input;
  const bool limiter_activated =
      minmod(make_not_null(&input_to_limit), mesh, element, logical_coords,
             element_size, neighbor_data);
  CHECK(limiter_activated);
  const ScalarTag::type expected_output = [&logical_coords, &mesh](
                                              const ScalarTag::type& in,
                                              const double slope) noexcept {
    const double mean = mean_value(get(in), mesh);
    return ScalarTag::type(mean + get<0>(logical_coords) * slope);
  }(input, expected_slope);
  CHECK_ITERABLE_APPROX(input_to_limit, expected_output);
}

// Make a 2D element with two neighbors in lower_xi, one neighbor in upper_xi.
// Check that lower_xi data from two neighbors is correctly combined in the
// limiting operation.
void test_minmod_limiter_two_lower_xi_neighbors() noexcept {
  const auto element = Element<2>{
      ElementId<2>{0},
      Element<2>::Neighbors_t{
          {Direction<2>::lower_xi(),
           {std::unordered_set<ElementId<2>>{ElementId<2>(1), ElementId<2>(7)},
            OrientationMap<2>{}}},
          {Direction<2>::upper_xi(),
           TestHelpers::Limiters::make_neighbor_with_id<2>(2)}}};
  const auto mesh =
      Mesh<2>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const double dx = 1.0;
  const auto element_size = make_array<2>(dx);

  const auto mean = 2.0;
  const auto func =
      [&mean](const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
        return mean + 1.2 * get<0>(coords);
      };
  const auto input = ScalarTag::type(func(logical_coords));

  const auto make_neighbors = [&dx](const double left1, const double left2,
                                    const double right, const double left1_size,
                                    const double left2_size) noexcept {
    using Pack = Limiters::Minmod<2, tmpl::list<ScalarTag>>::PackagedData;
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

  const double tvb_constant = 0.0;
  const Limiters::Minmod<2, tmpl::list<ScalarTag>> minmod(
      Limiters::MinmodType::LambdaPi1, tvb_constant);

  // Make two left neighbors with different mean values
  const auto neighbor_data_two_means =
      make_neighbors(mean - 1.1, mean - 1.0, mean + 1.4, dx, dx);
  // Effective neighbor mean (1.1 + 1.0) / 2.0 => 1.05
  test_limiter_activates_work(minmod, input, mesh, element, logical_coords,
                              element_size, neighbor_data_two_means, 1.05);

  // Make two left neighbors with different means and sizes
  const auto neighbor_data_two_sizes =
      make_neighbors(mean - 1.1, mean - 1.0, mean + 1.4, dx, 0.5 * dx);
  // Effective neighbor mean (1.1 + 1.0) / 2.0 => 1.05
  // Average neighbor size (1.0 + 0.5) / 2.0 => 0.75
  // Effective distance (0.75 + 1.0) / 2.0 => 0.875
  test_limiter_activates_work(minmod, input, mesh, element, logical_coords,
                              element_size, neighbor_data_two_sizes,
                              1.05 / 0.875);
}

// See above, but in 3D and with 4 upper_xi neighbors
void test_minmod_limiter_four_upper_xi_neighbors() noexcept {
  const auto element = Element<3>{
      ElementId<3>{0},
      Element<3>::Neighbors_t{
          {Direction<3>::lower_xi(),
           TestHelpers::Limiters::make_neighbor_with_id<3>(1)},
          {Direction<3>::upper_xi(),
           {std::unordered_set<ElementId<3>>{ElementId<3>(2), ElementId<3>(7),
                                             ElementId<3>(8), ElementId<3>(9)},
            OrientationMap<3>{}}},
      }};
  const auto mesh =
      Mesh<3>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const double dx = 1.0;
  const auto element_size = make_array<3>(dx);

  const auto mean = 2.0;
  const auto func =
      [&mean](const tnsr::I<DataVector, 3, Frame::Logical>& coords) noexcept {
        return mean + 1.2 * get<0>(coords);
      };
  const auto input = ScalarTag::type(func(logical_coords));

  const auto make_neighbors =
      [&dx](const double left, const double right1, const double right2,
            const double right3, const double right4, const double right1_size,
            const double right2_size, const double right3_size,
            const double right4_size) noexcept {
        using Pack = Limiters::Minmod<3, tmpl::list<ScalarTag>>::PackagedData;
        return std::unordered_map<
            std::pair<Direction<3>, ElementId<3>>, Pack,
            boost::hash<std::pair<Direction<3>, ElementId<3>>>>{
            std::make_pair(
                std::make_pair(Direction<3>::lower_xi(), ElementId<3>(1)),
                Pack{Scalar<double>(left), make_array<3>(dx)}),
            std::make_pair(
                std::make_pair(Direction<3>::upper_xi(), ElementId<3>(2)),
                Pack{Scalar<double>(right1), make_array(right1_size, dx, dx)}),
            std::make_pair(
                std::make_pair(Direction<3>::upper_xi(), ElementId<3>(7)),
                Pack{Scalar<double>(right2), make_array(right2_size, dx, dx)}),
            std::make_pair(
                std::make_pair(Direction<3>::upper_xi(), ElementId<3>(8)),
                Pack{Scalar<double>(right3), make_array(right3_size, dx, dx)}),
            std::make_pair(
                std::make_pair(Direction<3>::upper_xi(), ElementId<3>(9)),
                Pack{Scalar<double>(right4), make_array(right4_size, dx, dx)}),
        };
      };

  const double tvb_constant = 0.0;
  const Limiters::Minmod<3, tmpl::list<ScalarTag>> minmod(
      Limiters::MinmodType::LambdaPi1, tvb_constant);

  // Make four right neighbors with different mean values
  const auto neighbor_data_two_means =
      make_neighbors(mean - 1.4, mean + 1.0, mean + 1.1, mean - 0.2, mean + 1.8,
                     dx, dx, dx, dx);
  // Effective neighbor mean (1.0 + 1.1 - 0.2 + 1.8) / 4.0 => 0.925
  test_limiter_activates_work(minmod, input, mesh, element, logical_coords,
                              element_size, neighbor_data_two_means, 0.925);

  // Make four right neighbors with different means and sizes
  const auto neighbor_data_two_sizes =
      make_neighbors(mean - 1.4, mean + 1.0, mean + 1.1, mean - 0.2, mean + 1.8,
                     dx, 0.5 * dx, 0.5 * dx, 0.5 * dx);
  // Effective neighbor mean (1.0 + 1.1 - 0.2 + 1.8) / 4.0 => 0.925
  // Average neighbor size (1.0 + 0.5 + 0.5 + 0.5) / 4.0 => 0.625
  // Effective distance (0.625 + 1.0) / 2.0 => 0.8125
  test_limiter_activates_work(minmod, input, mesh, element, logical_coords,
                              element_size, neighbor_data_two_sizes,
                              0.925 / 0.8125);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.Minmod", "[Limiters][Unit]") {
  test_minmod_option_parsing();
  test_minmod_serialization();

  test_package_data_1d();
  test_package_data_2d();
  test_package_data_3d();

  // These functions test
  // - the TCI for the limiter, i.e., when the limiter activates
  // - the reduced slopes requested in the event of an activation
  test_minmod_limited_slopes_1d();
  test_minmod_limited_slopes_2d();
  test_minmod_limited_slopes_3d();

  // These functions test the correctness of the limited solution
  test_minmod_limiter_1d();
  test_minmod_limiter_2d();
  test_minmod_limiter_3d();

  // These functions test the limiter with h-refinement
  test_minmod_limiter_two_lower_xi_neighbors();
  test_minmod_limiter_four_upper_xi_neighbors();
}
