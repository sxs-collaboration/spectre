// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"  // IWYU pragma: keep
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

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

// Test that TCI detects a troubled cell when expected
template <size_t VolumeDim>
void test_tci_detection(
    const bool expected_detection, const double tvb_constant,
    const DataVector& input, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const DirectionMap<VolumeDim, double>& effective_neighbor_means,
    const DirectionMap<VolumeDim, double>& effective_neighbor_sizes) noexcept {
  Limiters::Minmod_detail::BufferWrapper<VolumeDim> buffer(mesh);
  const bool detection = Limiters::Tci::tvb_minmod_indicator(
      make_not_null(&buffer), tvb_constant, input, mesh, element, element_size,
      effective_neighbor_means, effective_neighbor_sizes);
  CHECK(detection == expected_detection);
}

void test_tci_on_linear_function(
    const size_t number_of_grid_points,
    const Spectral::Quadrature quadrature) noexcept {
  INFO("Testing linear function...");
  CAPTURE(number_of_grid_points);
  CAPTURE(quadrature);
  const auto element = TestHelpers::Limiters::make_element<1>();
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     quadrature);

  // Lambda takes tvb_scale = tvb_constant * h^2, to facilitate specifying
  // critical threshold values for testing
  const auto test_tci = [&mesh, &element](
                            const bool expected, const DataVector& input,
                            const double left_mean, const double right_mean,
                            const double tvb_scale) noexcept {
    const double h = 1.2;
    const double tvb_constant = tvb_scale / square(h);
    const auto element_size = make_array<1>(h);
    test_tci_detection(expected, tvb_constant, input, mesh, element,
                       element_size, make_two_neighbors(left_mean, right_mean),
                       make_two_neighbors(h, h));
  };

  // mean = 1.6
  // delta_left = delta_right = 0.2
  const DataVector input = [&mesh]() noexcept {
    const auto x = get<0>(logical_coordinates(mesh));
    return DataVector{1.6 + 0.2 * x};
  }();

  // Test trigger due to left, right neighbors
  test_tci(false, input, 1.35, 1.85, 0.0);
  test_tci(true, input, 1.45, 1.85, 0.0);
  test_tci(true, input, 1.35, 1.75, 0.0);
  test_tci(true, input, 1.45, 1.75, 0.0);

  // Test trigger due to slope changing sign
  test_tci(true, input, 1.7, 1.85, 0.0);
  test_tci(true, input, 1.45, 1.5, 0.0);

  // Test TVB can avoid the triggers
  test_tci(true, input, 1.45, 1.75, 0.19);
  test_tci(true, input, 1.7, 1.85, 0.19);
  test_tci(true, input, 1.45, 1.5, 0.19);
  test_tci(false, input, 1.45, 1.75, 0.21);
  test_tci(false, input, 1.7, 1.85, 0.21);
  test_tci(false, input, 1.45, 1.5, 0.21);
}

void test_tci_on_quadratic_function(
    const size_t number_of_grid_points,
    const Spectral::Quadrature quadrature) noexcept {
  INFO("Testing quadratic function...");
  CAPTURE(number_of_grid_points);
  CAPTURE(quadrature);
  const auto element = TestHelpers::Limiters::make_element<1>();
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     quadrature);

  // Lambda takes tvb_scale = tvb_constant * h^2, to facilitate specifying
  // critical threshold values for testing
  const auto test_tci = [&mesh, &element](
                            const bool expected, const DataVector& input,
                            const double left_mean, const double right_mean,
                            const double tvb_scale) noexcept {
    const double h = 1.2;
    const double tvb_constant = tvb_scale / square(h);
    const auto element_size = make_array<1>(h);
    test_tci_detection(expected, tvb_constant, input, mesh, element,
                       element_size, make_two_neighbors(left_mean, right_mean),
                       make_two_neighbors(h, h));
  };

  // mean = 1.5
  // delta_left = 0.1
  // delta_right = 0.3
  const DataVector input = [&mesh]() noexcept {
    const auto x = get<0>(logical_coordinates(mesh));
    return DataVector{1.45 + 0.2 * x + 0.15 * square(x)};
  }();

  // Test trigger due to left, right neighbors
  test_tci(false, input, 1.15, 1.85, 0.0);
  test_tci(true, input, 1.25, 1.85, 0.0);
  test_tci(true, input, 1.15, 1.75, 0.0);
  test_tci(true, input, 1.25, 1.75, 0.0);

  // Test TVB can avoid the trigger
  test_tci(true, input, 1.25, 1.85, 0.11);
  test_tci(true, input, 1.25, 1.85, 0.29);
  test_tci(false, input, 1.25, 1.85, 0.31);

  // mean = 1.5
  // delta_left = -0.1
  // delta_right = 0.3
  const DataVector input2 = [&mesh]() noexcept {
    const auto x = get<0>(logical_coordinates(mesh));
    return DataVector{1.4 + 0.1 * x + 0.3 * square(x)};
  }();

  // Because left-to-mean and mean-to-right slopes have different signs,
  // any TCI call with TVB=0 should trigger
  test_tci(true, input2, 1.7, 1.9, 0.0);
  test_tci(true, input2, 1.7, 1.3, 0.0);
  test_tci(true, input2, 1.3, 1.7, 0.0);
  test_tci(true, input2, 1.1, 1.9, 0.0);

  // Conversely, because left-to-mean and mean-to-right slopes have different
  // signs, calls with TVB>0 give results that depend on neighbor means
  test_tci(true, input2, 1.7, 1.3, 0.11);
  test_tci(true, input2, 1.7, 1.3, 0.29);
  test_tci(false, input2, 1.7, 1.3, 0.31);

  test_tci(true, input2, 1.3, 1.7, 0.11);
  test_tci(true, input2, 1.3, 1.7, 0.29);
  test_tci(false, input2, 1.3, 1.7, 0.31);

  test_tci(false, input2, 1.1, 1.9, 0.11);
  test_tci(false, input2, 1.1, 1.9, 0.29);
  test_tci(false, input2, 1.1, 1.9, 0.31);
}

void test_tci_at_boundary(const size_t number_of_grid_points,
                          const Spectral::Quadrature quadrature) noexcept {
  INFO("Testing limiter at boundary...");
  CAPTURE(number_of_grid_points);
  CAPTURE(quadrature);
  const double tvb_constant = 0.0;
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     quadrature);
  const auto element_size = make_array<1>(2.0);

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{1.2 * get<0>(coords)};
  }();

  // Test with element that has external lower-xi boundary
  const auto element_at_lower_xi_boundary =
      TestHelpers::Limiters::make_element<1>({{Direction<1>::lower_xi()}});
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_tci_detection(
        true, tvb_constant, input, mesh, element_at_lower_xi_boundary,
        element_size, {{std::make_pair(Direction<1>::upper_xi(), neighbor)}},
        {{std::make_pair(Direction<1>::upper_xi(), element_size[0])}});
  }

  // Test with element that has external upper-xi boundary
  const auto element_at_upper_xi_boundary =
      TestHelpers::Limiters::make_element<1>({{Direction<1>::upper_xi()}});
  for (const double neighbor : {-1.3, 3.6, 4.8, 13.2}) {
    test_tci_detection(
        true, tvb_constant, input, mesh, element_at_upper_xi_boundary,
        element_size, {{std::make_pair(Direction<1>::lower_xi(), neighbor)}},
        {{std::make_pair(Direction<1>::lower_xi(), element_size[0])}});
  }
}

void test_tci_with_different_size_neighbor(
    const size_t number_of_grid_points,
    const Spectral::Quadrature quadrature) noexcept {
  INFO("Testing limiter with neighboring elements of different size...");
  CAPTURE(number_of_grid_points);
  CAPTURE(quadrature);
  const double tvb_constant = 0.0;
  const auto element = TestHelpers::Limiters::make_element<1>();
  const Mesh<1> mesh(number_of_grid_points, Spectral::Basis::Legendre,
                     quadrature);
  const double dx = 1.0;
  const auto element_size = make_array<1>(dx);

  const auto test_tci = [&tvb_constant, &element, &mesh, &element_size](
                            const bool expected_detection,
                            const DataVector& local_input, const double left,
                            const double right, const double left_size,
                            const double right_size) noexcept {
    test_tci_detection(expected_detection, tvb_constant, local_input, mesh,
                       element, element_size, make_two_neighbors(left, right),
                       make_two_neighbors(left_size, right_size));
  };

  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    return DataVector{2.0 + 1.2 * get<0>(coords)};
  }();

  // Establish baseline using evenly-sized elements
  test_tci(false, input, 0.8 - eps, 3.2 + eps, dx, dx);

  const double larger = 2.0 * dx;
  const double smaller = 0.5 * dx;

  // Larger neighbor with same mean => true reduction in slope => trigger
  test_tci(true, input, 0.8 - eps, 3.2, dx, larger);
  // Larger neighbor with larger mean => same slope => no trigger
  test_tci(false, input, 0.8 - eps, 3.8 + eps, dx, larger);

  // Smaller neighbor with same mean => increased slope => no trigger
  test_tci(false, input, 0.8 - eps, 3.2 + eps, dx, smaller);
  // Smaller neighbor with lower mean => same slope => no trigger
  test_tci(false, input, 0.8 - eps, 2.9 + eps, dx, smaller);

  test_tci(true, input, 0.8, 3.2 + eps, larger, dx);
  test_tci(false, input, 0.2 - eps, 3.2 + eps, larger, dx);

  test_tci(false, input, 0.8 - eps, 3.2 + eps, smaller, dx);
  test_tci(false, input, 1.1 - eps, 3.2 + eps, smaller, dx);
}

// In 1D, test combinations of TVB constant, polynomial order, etc.
// Check that each combination has the expected TCI behavior.
void test_tvb_minmod_tci_1d() noexcept {
  INFO("Testing MinmodTci in 1D");
  for (const auto quadrature :
       {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
    for (const auto num_grid_points : std::array<size_t, 2>{{2, 4}}) {
      test_tci_on_linear_function(num_grid_points, quadrature);
      test_tci_at_boundary(num_grid_points, quadrature);
      test_tci_with_different_size_neighbor(num_grid_points, quadrature);
    }
    // This test only makes sense with more than 2 grid points
    test_tci_on_quadratic_function(3, quadrature);
    test_tci_on_quadratic_function(4, quadrature);
  }
}

void test_tvb_minmod_tci_2d_impl(
    const Spectral::Quadrature quadrature) noexcept {
  CAPTURE(quadrature);
  const double tvb_constant = 0.0;
  const auto element = TestHelpers::Limiters::make_element<2>();
  const Mesh<2> mesh(3, Spectral::Basis::Legendre, quadrature);
  const auto element_size = make_array<2>(2.0);

  const auto test_tci =
      [&tvb_constant, &element, &mesh, &element_size](
          const bool expected_detection, const DataVector& local_input,
          const std::array<double, 4>& neighbor_means) noexcept {
        test_tci_detection(expected_detection, tvb_constant, local_input, mesh,
                           element, element_size,
                           make_four_neighbors(neighbor_means),
                           make_four_neighbors(make_array<4>(2.0)));
      };

  const auto input = [&mesh]() noexcept {
    const auto coords = logical_coordinates(mesh);
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    return DataVector{3.0 + x + 2.0 * y + 0.1 * x * y};
  }();

  // Case with no activation
  test_tci(false, input, {{1.9, 4.2, -0.5, 5.6}});

  // Limit because of xi-direction neighbors
  test_tci(true, input, {{2.2, 4.2, -0.5, 5.6}});
  test_tci(true, input, {{1.9, 3.2, -0.5, 5.6}});

  // Limit because of eta-direction neighbors
  test_tci(true, input, {{1.9, 4.2, 1.5, 5.6}});
  test_tci(true, input, {{1.9, 4.2, -0.5, 2.9}});

  // Limit for xi and eta directions
  test_tci(true, input, {{2.2, 4.2, 1.5, 5.6}});
  test_tci(true, input, {{3.9, 4.2, -0.5, 2.9}});
}

// In 2D, test that the dimension-by-dimension application of the TCI works as
// expected.
void test_tvb_minmod_tci_2d() noexcept {
  INFO("Testing MinmodTci in 2D");
  test_tvb_minmod_tci_2d_impl(Spectral::Quadrature::GaussLobatto);
  test_tvb_minmod_tci_2d_impl(Spectral::Quadrature::Gauss);
}

void test_tvb_minmod_tci_3d_impl(
    const Spectral::Quadrature quadrature) noexcept {
  CAPTURE(quadrature);
  const double tvb_constant = 0.0;
  const auto element = TestHelpers::Limiters::make_element<3>();
  const Mesh<3> mesh(3, Spectral::Basis::Legendre, quadrature);
  const auto element_size = make_array<3>(2.0);

  const auto test_tci =
      [&tvb_constant, &element, &mesh, &element_size](
          const bool expected_detection, const DataVector& local_input,
          const std::array<double, 6>& neighbor_means) noexcept {
        test_tci_detection(expected_detection, tvb_constant, local_input, mesh,
                           element, element_size,
                           make_six_neighbors(neighbor_means),
                           make_six_neighbors(make_array<6>(2.0)));
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
  test_tci(false, input, {{3.8, -0.1, 1.5, 2.7, 1.2, 2.5}});

  // Limit because of xi-direction neighbors
  test_tci(true, input, {{3.4, -0.1, 1.5, 2.7, 1.2, 2.5}});
  test_tci(true, input, {{3.8, 2.1, 1.5, 2.7, 1.2, 2.5}});

  // Limit because of eta-direction neighbors
  test_tci(true, input, {{3.8, -0.1, 1.9, 2.7, 1.2, 2.5}});
  test_tci(true, input, {{3.8, -0.1, 1.5, 2.3, 1.2, 2.5}});

  // Limit because of zeta-direction neighbors
  test_tci(true, input, {{3.8, -0.1, 1.5, 2.7, 2.2, 2.5}});
  test_tci(true, input, {{3.8, -0.1, 1.5, 2.7, 1.2, 2.1}});

  // Limit for xi, eta, and zeta directions
  test_tci(true, input, {{3.4, -0.1, 1.5, 2.3, 1.2, 2.1}});
  test_tci(true, input, {{3.8, 2.1, 2.1, 2.7, 2.2, 2.5}});
}

// In 3D, test that the dimension-by-dimension application of the TCI works as
// expected.
void test_tvb_minmod_tci_3d() noexcept {
  INFO("Testing MinmodTci in 3D");
  test_tvb_minmod_tci_3d_impl(Spectral::Quadrature::GaussLobatto);
  test_tvb_minmod_tci_3d_impl(Spectral::Quadrature::Gauss);
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

void test_tvb_minmod_tci_several_tensors() noexcept {
  INFO("Testing MinmodTci action on several tensors");
  // Test that TCI returns true if just one component needs limiting, which
  // we do by limiting a scalar and vector in 3D
  const double tvb_constant = 0.0;
  const auto element = TestHelpers::Limiters::make_element<3>();
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
  const bool trigger_base_case =
      Limiters::Tci::tvb_minmod_indicator<3, TestPackagedData, ScalarTag,
                                          VectorTag<3>>(
          tvb_constant, local_scalar, local_vector, mesh, element, element_size,
          neighbor_data);
  CHECK_FALSE(trigger_base_case);

  // Case where the scalar triggers limiting
  get(get<::Tags::Mean<ScalarTag>>(upper_xi_neighbor.means)) = 2.0;
  const bool trigger_scalar =
      Limiters::Tci::tvb_minmod_indicator<3, TestPackagedData, ScalarTag,
                                          VectorTag<3>>(
          tvb_constant, local_scalar, local_vector, mesh, element, element_size,
          neighbor_data);
  CHECK(trigger_scalar);

  // Case where the vector x-component triggers limiting
  get(get<::Tags::Mean<ScalarTag>>(upper_xi_neighbor.means)) = 3.3;
  get<0>(get<::Tags::Mean<VectorTag<3>>>(lower_zeta_neighbor.means)) = -0.1;
  const bool trigger_vector_x =
      Limiters::Tci::tvb_minmod_indicator<3, TestPackagedData, ScalarTag,
                                          VectorTag<3>>(
          tvb_constant, local_scalar, local_vector, mesh, element, element_size,
          neighbor_data);
  CHECK(trigger_vector_x);

  // Case where the vector y-component triggers limiting
  get<0>(get<::Tags::Mean<VectorTag<3>>>(lower_zeta_neighbor.means)) = -1.8;
  get<1>(get<::Tags::Mean<VectorTag<3>>>(upper_eta_neighbor.means)) = -0.2;
  const bool trigger_vector_y =
      Limiters::Tci::tvb_minmod_indicator<3, TestPackagedData, ScalarTag,
                                          VectorTag<3>>(
          tvb_constant, local_scalar, local_vector, mesh, element, element_size,
          neighbor_data);
  CHECK(trigger_vector_y);

  // Case where the vector z-component triggers limiting
  get<1>(get<::Tags::Mean<VectorTag<3>>>(upper_eta_neighbor.means)) = 0.1;
  get<2>(get<::Tags::Mean<VectorTag<3>>>(lower_xi_neighbor.means)) = 1.9;
  const bool trigger_vector_z =
      Limiters::Tci::tvb_minmod_indicator<3, TestPackagedData, ScalarTag,
                                          VectorTag<3>>(
          tvb_constant, local_scalar, local_vector, mesh, element, element_size,
          neighbor_data);
  CHECK(trigger_vector_z);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.MinmodTci", "[Limiters][Unit]") {
  test_tvb_minmod_tci_1d();
  test_tvb_minmod_tci_2d();
  test_tvb_minmod_tci_3d();

  test_tvb_minmod_tci_several_tensors();
}
