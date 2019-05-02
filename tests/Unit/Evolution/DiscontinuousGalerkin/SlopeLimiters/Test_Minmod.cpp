// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.tpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
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

// IWYU pragma: no_include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"
// IWYU pragma: no_forward_declare SlopeLimiters::Minmod
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
  const auto lambda_pi1_default =
      test_creation<SlopeLimiters::Minmod<1, tmpl::list<ScalarTag>>>(
          "  Type: LambdaPi1");
  const auto lambda_pi1_m0 =
      test_creation<SlopeLimiters::Minmod<1, tmpl::list<ScalarTag>>>(
          "  Type: LambdaPi1\n  TvbmConstant: 0.0");
  const auto lambda_pi1_m1 =
      test_creation<SlopeLimiters::Minmod<1, tmpl::list<ScalarTag>>>(
          "  Type: LambdaPi1\n  TvbmConstant: 1.0");
  const auto muscl_default =
      test_creation<SlopeLimiters::Minmod<1, tmpl::list<ScalarTag>>>(
          "  Type: Muscl");

  // Test default TVBM value, operator==, and operator!=
  CHECK(lambda_pi1_default == lambda_pi1_m0);
  CHECK(lambda_pi1_default != lambda_pi1_m1);
  CHECK(lambda_pi1_default != muscl_default);

  test_creation<SlopeLimiters::Minmod<1, tmpl::list<ScalarTag>>>(
      "  Type: LambdaPiN");
  test_creation<SlopeLimiters::Minmod<2, tmpl::list<ScalarTag>>>(
      "  Type: LambdaPiN");
  test_creation<SlopeLimiters::Minmod<3, tmpl::list<ScalarTag, VectorTag<3>>>>(
      "  Type: LambdaPiN");

  test_creation<SlopeLimiters::Minmod<3, tmpl::list<ScalarTag>>>(
      "  Type: LambdaPiN\n  DisableForDebugging: True");
}

void test_minmod_serialization() noexcept {
  const SlopeLimiters::Minmod<1, tmpl::list<ScalarTag>> minmod(
      SlopeLimiters::MinmodType::LambdaPi1);
  test_serialization(minmod);
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

// Test that the limiter activates in the x-direction only. Domain quantities
// and input Scalar may be of higher dimension VolumeDim.
template <size_t VolumeDim>
void test_limiter_activates_work(
    const SlopeLimiters::Minmod<VolumeDim, tmpl::list<ScalarTag>>& minmod,
    const ScalarTag::type& input, const Element<VolumeDim>& element,
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename SlopeLimiters::Minmod<VolumeDim,
                                       tmpl::list<ScalarTag>>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const double expected_slope) noexcept {
  auto input_to_limit = input;
  const bool limiter_activated =
      minmod(make_not_null(&input_to_limit), element, mesh, logical_coords,
             element_size, neighbor_data);
  CHECK(limiter_activated);
  const ScalarTag::type expected_output = [&logical_coords, &mesh ](
      const ScalarTag::type& in, const double slope) noexcept {
    const double mean = mean_value(get(in), mesh);
    return ScalarTag::type(mean + get<0>(logical_coords) * slope);
  }
  (input, expected_slope);
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
  const auto input = ScalarTag::type(func(logical_coords));

  const auto make_neighbors = [&dx](const double left1, const double left2,
                                    const double right, const double left1_size,
                                    const double left2_size) noexcept {
    using Pack = SlopeLimiters::Minmod<2, tmpl::list<ScalarTag>>::PackagedData;
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

  const SlopeLimiters::Minmod<2, tmpl::list<ScalarTag>> minmod(
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

// See above, but in 3D and with 4 upper_xi neighbors
void test_minmod_limiter_four_upper_xi_neighbors() noexcept {
  const auto element = Element<3>{
      ElementId<3>{0},
      Element<3>::Neighbors_t{
          {Direction<3>::lower_xi(), make_neighbor_with_id<3>(1)},
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
  const auto func = [&mean](
      const tnsr::I<DataVector, 3, Frame::Logical>& coords) noexcept {
    return mean + 1.2 * get<0>(coords);
  };
  const auto input = ScalarTag::type(func(logical_coords));

  const auto make_neighbors = [&dx](
      const double left, const double right1, const double right2,
      const double right3, const double right4, const double right1_size,
      const double right2_size, const double right3_size,
      const double right4_size) noexcept {
    using Pack = SlopeLimiters::Minmod<3, tmpl::list<ScalarTag>>::PackagedData;
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

  const SlopeLimiters::Minmod<3, tmpl::list<ScalarTag>> minmod(
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

  const SlopeLimiters::Minmod<VolumeDim,
                              tmpl::list<ScalarTag, VectorTag<VolumeDim>>>
      minmod(SlopeLimiters::MinmodType::LambdaPi1);
  typename SlopeLimiters::Minmod<
      VolumeDim, tmpl::list<ScalarTag, VectorTag<VolumeDim>>>::PackagedData
      packaged_data{};

  // First we test package_data with an identity orientation_map
  minmod.package_data(make_not_null(&packaged_data), input_scalar,
                      modified_vector, mesh, element_size, {});

  // Should not normally look inside the package, but we do so here for testing.
  double lhs = get(get<Tags::Mean<ScalarTag>>(packaged_data.means));
  CHECK(lhs == approx(mean_value(get(input_scalar), mesh)));
  for (size_t d = 0; d < VolumeDim; ++d) {
    lhs = get<Tags::Mean<VectorTag<VolumeDim>>>(packaged_data.means).get(d);
    CHECK(lhs == approx(mean_value(modified_vector.get(d), mesh)));
  }
  CHECK(packaged_data.element_size == element_size);

  // Then we test with a reorientation, as if sending the data to another Block
  minmod.package_data(make_not_null(&packaged_data), input_scalar,
                      modified_vector, mesh, element_size, orientation_map);
  lhs = get(get<Tags::Mean<ScalarTag>>(packaged_data.means));
  CHECK(lhs == approx(mean_value(get(input_scalar), mesh)));
  for (size_t d = 0; d < VolumeDim; ++d) {
    lhs = get<Tags::Mean<VectorTag<VolumeDim>>>(packaged_data.means).get(d);
    CHECK(lhs == approx(mean_value(modified_vector.get(d), mesh)));
  }
  CHECK(packaged_data.element_size ==
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
            VolumeDim,
            tmpl::list<ScalarTag, VectorTag<VolumeDim>>>::PackagedData,
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
  const SlopeLimiters::Minmod<VolumeDim,
                              tmpl::list<ScalarTag, VectorTag<VolumeDim>>>
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

void test_minmod_limiter_1d() noexcept {
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
  const auto input_scalar = ScalarTag::type{data};
  const auto input_vector = VectorTag<1>::type{data};

  const OrientationMap<1> test_reorientation(
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
  test_package_data_work(input_scalar, input_vector, mesh, logical_coords,
                         element_size, test_reorientation);

  // b. Generate neighbor data for the scalar and vector Tensors.
  std::unordered_map<std::pair<Direction<1>, ElementId<1>>,
                     SlopeLimiters::Minmod<
                         1, tmpl::list<ScalarTag, VectorTag<1>>>::PackagedData,
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

  test_work(input_scalar, input_vector, neighbor_data, mesh, logical_coords,
            element_size, target_scalar_slope, target_vector_slope);
}

void test_minmod_limiter_2d() noexcept {
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
  const auto input_scalar = ScalarTag::type{data};
  const auto input_vector = VectorTag<2>::type{data};

  const OrientationMap<2> test_reorientation(std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}});
  test_package_data_work(input_scalar, input_vector, mesh, logical_coords,
                         element_size, test_reorientation);

  // b. Generate neighbor data for the scalar and vector Tensors.
  std::unordered_map<std::pair<Direction<2>, ElementId<2>>,
                     SlopeLimiters::Minmod<
                         2, tmpl::list<ScalarTag, VectorTag<2>>>::PackagedData,
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

  test_work(input_scalar, input_vector, neighbor_data, mesh, logical_coords,
            element_size, target_scalar_slope, target_vector_slope);
}

void test_minmod_limiter_3d() noexcept {
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
  const auto input_scalar = ScalarTag::type{data};
  const auto input_vector = VectorTag<3>::type{data};

  const OrientationMap<3> test_reorientation(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::lower_zeta()}});
  test_package_data_work(input_scalar, input_vector, mesh, logical_coords,
                         element_size, test_reorientation);

  // b. Generate neighbor data for the scalar and vector Tensors.
  std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                     SlopeLimiters::Minmod<
                         3, tmpl::list<ScalarTag, VectorTag<3>>>::PackagedData,
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

  test_work(input_scalar, input_vector, neighbor_data, mesh, logical_coords,
            element_size, target_scalar_slope, target_vector_slope);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.Minmod",
                  "[SlopeLimiters][Unit]") {
  {
    INFO("Test Minmod option-parsing and serialization");
    test_minmod_option_parsing();
    test_minmod_serialization();
  }

  {
    INFO("Test Minmod limiter in 1d");
    test_minmod_limiter_1d();
  }

  {
    INFO("Test Minmod limiter in 2d");
    test_minmod_limiter_2d();
    test_minmod_limiter_two_lower_xi_neighbors();
  }

  {
    INFO("Test Minmod limiter in 3d");
    test_minmod_limiter_3d();
    test_minmod_limiter_four_upper_xi_neighbors();
  }
}
