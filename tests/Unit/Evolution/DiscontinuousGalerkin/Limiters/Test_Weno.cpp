// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoModifiedSolution.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_include "DataStructures/DataBox/Prefixes.hpp"
// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"
// IWYU pragma: no_forward_declare Limiters::Weno
// IWYU pragma: no_forward_declare Tags::Mean
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

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

void test_weno_option_parsing() noexcept {
  INFO("Test WENO option parsing");

  const auto hweno_1d =
      TestHelpers::test_creation<Limiters::Weno<1, tmpl::list<ScalarTag>>>(
          "Type: Hweno");
  const auto hweno_1d_default_weight =
      TestHelpers::test_creation<Limiters::Weno<1, tmpl::list<ScalarTag>>>(
          "Type: Hweno\n"
          "NeighborWeight: 0.001");
  const auto hweno_1d_larger_weight =
      TestHelpers::test_creation<Limiters::Weno<1, tmpl::list<ScalarTag>>>(
          "Type: Hweno\n"
          "NeighborWeight: 0.01");
  const auto hweno_1d_disabled =
      TestHelpers::test_creation<Limiters::Weno<1, tmpl::list<ScalarTag>>>(
          "Type: Hweno\n"
          "DisableForDebugging: True");
  const auto simple_weno_1d =
      TestHelpers::test_creation<Limiters::Weno<1, tmpl::list<ScalarTag>>>(
          "Type: SimpleWeno");

  // Check neighbor_weight default from options, op==, op!=
  CHECK(hweno_1d == hweno_1d_default_weight);
  CHECK(hweno_1d != hweno_1d_larger_weight);
  CHECK(hweno_1d != hweno_1d_disabled);
  CHECK(hweno_1d != simple_weno_1d);

  const auto hweno_2d =
      TestHelpers::test_creation<Limiters::Weno<2, tmpl::list<ScalarTag>>>(
          "Type: Hweno");
  const auto hweno_3d_larger_weight = TestHelpers::test_creation<
      Limiters::Weno<3, tmpl::list<ScalarTag, VectorTag<3>>>>(
      "Type: Hweno\n"
      "NeighborWeight: 0.01\n"
      "DisableForDebugging: True");

  // Check that creation from options gives correct object
  const Limiters::Weno<1, tmpl::list<ScalarTag>> expected_hweno_1d(
      Limiters::WenoType::Hweno, 0.001);
  const Limiters::Weno<1, tmpl::list<ScalarTag>>
      expected_hweno_1d_larger_weight(Limiters::WenoType::Hweno, 0.01);
  const Limiters::Weno<1, tmpl::list<ScalarTag>> expected_hweno_1d_disabled(
      Limiters::WenoType::Hweno, 0.001, true);
  const Limiters::Weno<1, tmpl::list<ScalarTag>> expected_simple_weno_1d(
      Limiters::WenoType::SimpleWeno, 0.001);
  const Limiters::Weno<2, tmpl::list<ScalarTag>> expected_hweno_2d(
      Limiters::WenoType::Hweno, 0.001);
  const Limiters::Weno<3, tmpl::list<ScalarTag, VectorTag<3>>>
      expected_hweno_3d_larger_weight(Limiters::WenoType::Hweno, 0.01, true);
  CHECK(hweno_1d == expected_hweno_1d);
  CHECK(hweno_1d_larger_weight == expected_hweno_1d_larger_weight);
  CHECK(hweno_1d_disabled == expected_hweno_1d_disabled);
  CHECK(simple_weno_1d == expected_simple_weno_1d);
  CHECK(hweno_2d == expected_hweno_2d);
  CHECK(hweno_3d_larger_weight == expected_hweno_3d_larger_weight);
}

void test_weno_serialization() noexcept {
  INFO("Test WENO serialization");
  const Limiters::Weno<1, tmpl::list<ScalarTag>> weno(Limiters::WenoType::Hweno,
                                                      0.01, true);
  test_serialization(weno);
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
  const Limiters::Weno<VolumeDim, TagList> weno(Limiters::WenoType::Hweno,
                                                0.001);
  typename Limiters::Weno<VolumeDim, TagList>::PackagedData packaged_data{};

  weno.package_data(make_not_null(&packaged_data), input_scalar, input_vector,
                    mesh, element_size, orientation_map);

  const Variables<TagList> oriented_vars =
      [&mesh, &input_scalar, &input_vector, &orientation_map ]() noexcept {
    Variables<TagList> input_vars(mesh.number_of_grid_points());
    get<ScalarTag>(input_vars) = input_scalar;
    get<VectorTag<VolumeDim>>(input_vars) = input_vector;
    return orient_variables(input_vars, mesh.extents(), orientation_map);
  }
  ();
  CHECK(packaged_data.volume_data == oriented_vars);
  CHECK(packaged_data.mesh == orientation_map(mesh));
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
  INFO("Test WENO package_data in 1D");
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

template <size_t VolumeDim>
void test_weno_tci_work(
    const Mesh<VolumeDim>& mesh, const Scalar<DataVector>& scalar,
    const std::array<double, 2 * VolumeDim>& means_no_activation,
    const std::array<double, 2 * VolumeDim>& means_activation) noexcept {
  using Weno = Limiters::Weno<VolumeDim, tmpl::list<ScalarTag>>;
  const auto element = TestHelpers::Limiters::make_element<VolumeDim>();
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<VolumeDim>(1.2);

  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      typename Weno::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};
  for (size_t d = 0; d < VolumeDim; ++d) {
    for (const auto side : {Side::Lower, Side::Upper}) {
      const size_t index = 2 * d + (side == Side::Lower ? 0 : 1);
      const auto dir_and_id = std::make_pair(Direction<VolumeDim>(d, side),
                                             ElementId<VolumeDim>(1 + index));
      neighbor_data[dir_and_id].volume_data =
          Variables<tmpl::list<ScalarTag>>(mesh.number_of_grid_points(), 0.);
      neighbor_data[dir_and_id].means =
          tuples::TaggedTuple<::Tags::Mean<ScalarTag>>(
              gsl::at(means_no_activation, index));
      neighbor_data[dir_and_id].mesh = mesh;
      neighbor_data[dir_and_id].element_size = element_size;
    }
  }

  const double neighbor_linear_weight = 0.001;
  const Weno sweno(Limiters::WenoType::SimpleWeno, neighbor_linear_weight);
  auto scalar_to_limit = scalar;
  bool activated = sweno(make_not_null(&scalar_to_limit), element, mesh,
                         element_size, neighbor_data);
  CHECK_FALSE(activated);
  CHECK(scalar_to_limit == scalar);

  for (size_t d = 0; d < VolumeDim; ++d) {
    for (const auto side : {Side::Lower, Side::Upper}) {
      const size_t index = 2 * d + (side == Side::Lower ? 0 : 1);
      const auto dir_and_id = std::make_pair(Direction<VolumeDim>(d, side),
                                             ElementId<VolumeDim>(1 + index));
      neighbor_data[dir_and_id].means =
          tuples::TaggedTuple<::Tags::Mean<ScalarTag>>(
              gsl::at(means_activation, index));
    }
  }
  activated = sweno(make_not_null(&scalar_to_limit), element, mesh,
                    element_size, neighbor_data);
  CHECK(activated);
  CHECK(scalar_to_limit != scalar);
}

void test_weno_tci_1d() noexcept {
  INFO("Test WENO's troubled-cell indicator in 1D");
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);

  const auto scalar = [&mesh]() noexcept {
    const auto logical_coords = logical_coordinates(mesh);
    const auto& x = get<0>(logical_coords);
    return ScalarTag::type{{{1.0 - 2.0 * x + square(x)}}};
  }
  ();

  // Here we specify two sets of neighbor means: one such that the TCI does not
  // trigger, another such that the TCI does trigger.
  // Local mean: 4/3; largest mean-to-edge slope: 8/3
  const std::array<double, 2> means_no_activation = {{4., -1.4}};
  const std::array<double, 2> means_activation = {{4., -1.3}};

  test_weno_tci_work(mesh, scalar, means_no_activation, means_activation);
}

void test_weno_tci_2d() noexcept {
  INFO("Test WENO's troubled-cell indicator in 2D");
  const auto mesh = Mesh<2>({{3, 3}}, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);

  const auto scalar = [&mesh]() noexcept {
    const auto logical_coords = logical_coordinates(mesh);
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    return ScalarTag::type{{{x + y - 0.5 * square(x) + 0.5 * square(y)}}};
  }
  ();

  // Local mean: 0; largest mean-to-edge slope in x and y: 4/3
  const std::array<double, 4> means_no_activation = {{-1.4, 1.4, -1.4, 1.4}};
  const std::array<double, 4> means_activation = {{-1.4, 1.4, -1.4, 1.3}};

  test_weno_tci_work(mesh, scalar, means_no_activation, means_activation);
}

void test_weno_tci_3d() noexcept {
  INFO("Test WENO's troubled-cell indicator in 3D");
  const auto mesh = Mesh<3>({{3, 4, 5}}, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);

  const auto scalar = [&mesh]() noexcept {
    const auto logical_coords = logical_coordinates(mesh);
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto& z = get<2>(logical_coords);
    return ScalarTag::type{{{x + y - 0.2 * z - y * z + x * y * square(z)}}};
  }
  ();

  // Local mean: 0; largest mean-to-edge slope in x and y: 1, in z: -0.2
  const std::array<double, 6> means_no_activation = {
      {-1.1, 1.1, -1.1, 1.1, 0.25, -0.25}};
  const std::array<double, 6> means_activation = {
      {-1.1, 1.1, -1.1, 1.1, 0.15, -0.25}};

  test_weno_tci_work(mesh, scalar, means_no_activation, means_activation);
}

template <size_t VolumeDim>
using VariablesMap = std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
    Variables<tmpl::list<ScalarTag, VectorTag<VolumeDim>>>,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>;

template <size_t VolumeDim>
std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
    typename Limiters::Weno<
        VolumeDim, tmpl::list<ScalarTag, VectorTag<VolumeDim>>>::PackagedData,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
make_neighbor_data_from_neighbor_vars(
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const VariablesMap<VolumeDim>& neighbor_vars) noexcept {
  const auto make_tuple_of_means = [&mesh](
      const Variables<tmpl::list<ScalarTag, VectorTag<VolumeDim>>>&
          vars_to_average) noexcept {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>,
                        ::Tags::Mean<VectorTag<VolumeDim>>>
        result;
    get(get<::Tags::Mean<ScalarTag>>(result)) =
        mean_value(get(get<ScalarTag>(vars_to_average)), mesh);
    for (size_t d = 0; d < VolumeDim; ++d) {
      get<::Tags::Mean<VectorTag<VolumeDim>>>(result).get(d) =
          mean_value(get<VectorTag<VolumeDim>>(vars_to_average).get(d), mesh);
    }
    return result;
  };

  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      typename Limiters::Weno<
          VolumeDim, tmpl::list<ScalarTag, VectorTag<VolumeDim>>>::PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};

  for (const auto& neighbor : element.neighbors()) {
    const auto dir = neighbor.first;
    const auto id = *(neighbor.second.cbegin());
    const auto dir_and_id = std::make_pair(dir, id);
    neighbor_data[dir_and_id].volume_data = neighbor_vars.at(dir_and_id);
    neighbor_data[dir_and_id].means =
        make_tuple_of_means(neighbor_vars.at(dir_and_id));
    neighbor_data[dir_and_id].mesh = mesh;
    neighbor_data[dir_and_id].element_size = element_size;
  }

  return neighbor_data;
}

template <size_t VolumeDim>
void test_weno_work(
    const Limiters::WenoType& weno_type,
    const Limiters::Weno_detail::DerivativeWeight derivative_weight,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const Variables<tmpl::list<ScalarTag, VectorTag<VolumeDim>>>& local_vars,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        typename Limiters::Weno<
            VolumeDim,
            tmpl::list<ScalarTag, VectorTag<VolumeDim>>>::PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data,
    const VariablesMap<VolumeDim>& expected_neighbor_modified_vars,
    Approx local_approx = approx) noexcept {
  // First run some sanity checks on the input, and make sure the test function
  // is being called in a reasonable way
  if (element.neighbors().size() != neighbor_data.size()) {
    ERROR("Different number of neighbors from element, neighbor_data");
  }
  if (neighbor_data.size() != expected_neighbor_modified_vars.size()) {
    ERROR("Different sizes for neighbor_data, expected_neighbor_modified_vars");
  }
  for (const auto& neighbor : element.neighbors()) {
    if (neighbor.second.ids().size() > 1) {
      ERROR("Too many neighbors: h-refinement is not yet supported");
    }
    const auto dir = neighbor.first;
    const auto id = *(neighbor.second.cbegin());
    const auto dir_and_id = std::make_pair(dir, id);
    if (neighbor_data.find(dir_and_id) == neighbor_data.end()) {
      ERROR("Missing neighbor_data at an internal boundary");
    }
    if (expected_neighbor_modified_vars.find(dir_and_id) ==
        expected_neighbor_modified_vars.end()) {
      ERROR("Missing expected_neighbor_modified_vars at an internal boundary");
    }
  }

  const double neighbor_linear_weight = 0.001;
  using Weno =
      Limiters::Weno<VolumeDim, tmpl::list<ScalarTag, VectorTag<VolumeDim>>>;

  auto scalar = get<ScalarTag>(local_vars);
  auto vector = get<VectorTag<VolumeDim>>(local_vars);

  // WENO should preserve the mean, so expected = initial
  const double expected_scalar_mean = mean_value(get(scalar), mesh);
  const auto expected_vector_means = [&vector, &mesh ]() noexcept {
    std::array<double, VolumeDim> means{};
    for (size_t d = 0; d < VolumeDim; ++d) {
      gsl::at(means, d) = mean_value(vector.get(d), mesh);
    }
    return means;
  }
  ();

  const Weno weno(weno_type, neighbor_linear_weight);
  const bool activated = weno(make_not_null(&scalar), make_not_null(&vector),
                              element, mesh, element_size, neighbor_data);

  CHECK(activated);

  CHECK(mean_value(get(scalar), mesh) == approx(expected_scalar_mean));
  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK(mean_value(vector.get(d), mesh) ==
          approx(gsl::at(expected_vector_means, d)));
  }

  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      expected_neighbor_polynomials;

  auto expected_scalar = get<ScalarTag>(local_vars);
  for (auto& neighbor_and_vars : expected_neighbor_modified_vars) {
    expected_neighbor_polynomials[neighbor_and_vars.first] =
        get(get<ScalarTag>(neighbor_and_vars.second));
  }
  Limiters::Weno_detail::reconstruct_from_weighted_sum(
      make_not_null(&get(expected_scalar)), mesh, neighbor_linear_weight,
      expected_neighbor_polynomials, derivative_weight);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_scalar, scalar, local_approx);

  auto expected_vector = get<VectorTag<VolumeDim>>(local_vars);
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (auto& neighbor_and_vars : expected_neighbor_modified_vars) {
      expected_neighbor_polynomials[neighbor_and_vars.first] =
          get<VectorTag<VolumeDim>>(neighbor_and_vars.second).get(i);
    }
    Limiters::Weno_detail::reconstruct_from_weighted_sum(
        make_not_null(&(expected_vector.get(i))), mesh, neighbor_linear_weight,
        expected_neighbor_polynomials, derivative_weight);
  }
  CHECK_ITERABLE_CUSTOM_APPROX(expected_vector, vector, local_approx);
}

void test_simple_weno_1d(const std::unordered_set<Direction<1>>&
                             directions_of_external_boundaries = {}) noexcept {
  INFO("Test simple WENO limiter in 1D");
  CAPTURE(directions_of_external_boundaries);
  const auto element =
      TestHelpers::Limiters::make_element<1>(directions_of_external_boundaries);
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(1.2);

  // Functions to produce dummy data on each element
  const auto make_center_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(x.size());
    get(get<ScalarTag>(vars)) = 1.0 - 2.0 * x + square(x);
    get<0>(get<VectorTag<1>>(vars)) = 0.4 * x - 0.1 * square(x);
    return vars;
  };
  const auto make_lower_xi_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords,
         const double offset = 0.0) noexcept {
    const auto x = get<0>(coords) + offset;
    Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(x.size());
    get(get<ScalarTag>(vars)) = -2.0 - 10.0 * x - square(x);
    get<0>(get<VectorTag<1>>(vars)) = -0.1 + 0.3 * x - 0.1 * square(x);
    return vars;
  };
  const auto make_upper_xi_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords,
         const double offset = 0.0) noexcept {
    const auto x = get<0>(coords) + offset;
    Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(x.size());
    get(get<ScalarTag>(vars)) = -0.3 - x + 0.5 * square(x);
    get<0>(get<VectorTag<1>>(vars)) = 0.6 * x - 0.3 * square(x);
    return vars;
  };

  const auto local_vars = make_center_vars(logical_coords);
  VariablesMap<1> neighbor_vars{};
  VariablesMap<1> neighbor_modified_vars{};

  const auto shift_vars_to_local_means = [&mesh, &local_vars ](
      const Variables<tmpl::list<ScalarTag, VectorTag<1>>>& input) noexcept {
    const auto& local_s = get<ScalarTag>(local_vars);
    const auto& local_v = get<VectorTag<1>>(local_vars);
    auto result = input;
    auto& s = get<ScalarTag>(result);
    auto& v = get<VectorTag<1>>(result);
    get(s) += mean_value(get(local_s), mesh) - mean_value(get(s), mesh);
    get<0>(v) +=
        mean_value(get<0>(local_v), mesh) - mean_value(get<0>(v), mesh);
    return result;
  };

  const auto lower_xi =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1));
  if (directions_of_external_boundaries.count(lower_xi.first) == 0) {
    neighbor_vars[lower_xi] = make_lower_xi_vars(logical_coords, -2.0);
    neighbor_modified_vars[lower_xi] =
        shift_vars_to_local_means(make_lower_xi_vars(logical_coords));
  }

  const auto upper_xi =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2));
  if (directions_of_external_boundaries.count(upper_xi.first) == 0) {
    neighbor_vars[upper_xi] = make_upper_xi_vars(logical_coords, 2.0);
    neighbor_modified_vars[upper_xi] =
        shift_vars_to_local_means(make_upper_xi_vars(logical_coords));
  }

  const auto neighbor_data = make_neighbor_data_from_neighbor_vars(
      element, mesh, element_size, neighbor_vars);

  test_weno_work<1>(Limiters::WenoType::SimpleWeno,
                    Limiters::Weno_detail::DerivativeWeight::PowTwoEll, element,
                    mesh, element_size, local_vars, neighbor_data,
                    neighbor_modified_vars);
}

void test_simple_weno_2d(const std::unordered_set<Direction<2>>&
                             directions_of_external_boundaries = {}) noexcept {
  INFO("Test simple WENO limiter in 2D");
  CAPTURE(directions_of_external_boundaries);
  const auto element =
      TestHelpers::Limiters::make_element<2>(directions_of_external_boundaries);
  const auto mesh =
      Mesh<2>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<2>(1.2);

  const auto make_center_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(x.size());
    get(get<ScalarTag>(vars)) = x + y - 0.5 * square(x) + 0.5 * square(y);
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) =
        0.1 + 0.2 * x - 0.4 * y + 0.3 * square(x) * square(y);
    return vars;
  };
  const auto make_lower_xi_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords,
         const double xi_offset = 0.0) noexcept {
    const auto x = get<0>(coords) + xi_offset;
    const auto& y = get<1>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(x.size());
    get(get<ScalarTag>(vars)) = 2.0 * x + 1.2 * y;
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) = 3.0 + 0.2 * y;
    return vars;
  };
  const auto make_upper_xi_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords,
         const double xi_offset = 0.0) noexcept {
    const auto x = get<0>(coords) + xi_offset;
    const auto& y = get<1>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(x.size());
    get(get<ScalarTag>(vars)) = x + y - 0.25 * square(x) - square(y);
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) = -2.4 + square(x);
    return vars;
  };
  const auto make_lower_eta_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords,
         const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto y = get<1>(coords) + eta_offset;
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(x.size());
    get(get<ScalarTag>(vars)) = -1 + 0.5 * x + y;
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) = 0.2 - y;
    return vars;
  };
  const auto make_upper_eta_vars =
      [](const tnsr::I<DataVector, 2, Frame::Logical>& coords,
         const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto y = get<1>(coords) + eta_offset;
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(x.size());
    get(get<ScalarTag>(vars)) =
        -6.0 + x + 2.0 * y + 0.5 * square(x) + 0.5 * square(y);
    get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
    get<1>(get<VectorTag<2>>(vars)) = 0.4 + 0.3 * x * square(y);
    return vars;
  };

  const auto local_vars = make_center_vars(logical_coords);
  VariablesMap<2> neighbor_vars{};
  VariablesMap<2> neighbor_modified_vars{};

  const auto shift_vars_to_local_means = [&mesh, &local_vars ](
      const Variables<tmpl::list<ScalarTag, VectorTag<2>>>& input) noexcept {
    const auto& local_s = get<ScalarTag>(local_vars);
    const auto& local_v = get<VectorTag<2>>(local_vars);
    auto result = input;
    auto& s = get<ScalarTag>(result);
    auto& v = get<VectorTag<2>>(result);
    get(s) += mean_value(get(local_s), mesh) - mean_value(get(s), mesh);
    get<0>(v) +=
        mean_value(get<0>(local_v), mesh) - mean_value(get<0>(v), mesh);
    get<1>(v) +=
        mean_value(get<1>(local_v), mesh) - mean_value(get<1>(v), mesh);
    return result;
  };

  const auto lower_xi =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>(1));
  if (directions_of_external_boundaries.count(lower_xi.first) == 0) {
    neighbor_vars[lower_xi] = make_lower_xi_vars(logical_coords, -2.0);
    neighbor_modified_vars[lower_xi] =
        shift_vars_to_local_means(make_lower_xi_vars(logical_coords));
  }

  const auto upper_xi =
      std::make_pair(Direction<2>::upper_xi(), ElementId<2>(2));
  if (directions_of_external_boundaries.count(upper_xi.first) == 0) {
    neighbor_vars[upper_xi] = make_upper_xi_vars(logical_coords, 2.0);
    neighbor_modified_vars[upper_xi] =
        shift_vars_to_local_means(make_upper_xi_vars(logical_coords));
  }

  const auto lower_eta =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>(3));
  if (directions_of_external_boundaries.count(lower_eta.first) == 0) {
    neighbor_vars[lower_eta] = make_lower_eta_vars(logical_coords, -2.0);
    neighbor_modified_vars[lower_eta] =
        shift_vars_to_local_means(make_lower_eta_vars(logical_coords));
  }

  const auto upper_eta =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>(4));
  if (directions_of_external_boundaries.count(upper_eta.first) == 0) {
    neighbor_vars[upper_eta] = make_upper_eta_vars(logical_coords, 2.0);
    neighbor_modified_vars[upper_eta] =
        shift_vars_to_local_means(make_upper_eta_vars(logical_coords));
  }

  const auto neighbor_data = make_neighbor_data_from_neighbor_vars(
      element, mesh, element_size, neighbor_vars);

  test_weno_work<2>(Limiters::WenoType::SimpleWeno,
                    Limiters::Weno_detail::DerivativeWeight::PowTwoEll, element,
                    mesh, element_size, local_vars, neighbor_data,
                    neighbor_modified_vars);
}

void test_simple_weno_3d(const std::unordered_set<Direction<3>>&
                             directions_of_external_boundaries = {}) noexcept {
  INFO("Test simple WENO limiter in 3D");
  CAPTURE(directions_of_external_boundaries);
  const auto element =
      TestHelpers::Limiters::make_element<3>(directions_of_external_boundaries);
  const auto mesh = Mesh<3>({{3, 4, 5}}, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<3>(1.2);

  const auto make_center_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(x.size());
    get(get<ScalarTag>(vars)) = x + y - 0.2 * z - y * z + x * y * square(z);
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = z;
    get<2>(get<VectorTag<3>>(vars)) = x + square(y) + cube(z);
    return vars;
  };
  const auto make_lower_xi_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double xi_offset = 0.0) noexcept {
    const auto x = get<0>(coords) + xi_offset;
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(x.size());
    get(get<ScalarTag>(vars)) = 1.2 * x + y - 0.4 * z + x * y * square(z);
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = 0.8 * z + 0.3 * x * y;
    get<2>(get<VectorTag<3>>(vars)) = x + y;
    return vars;
  };
  const auto make_upper_xi_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double xi_offset = 0.0) noexcept {
    const auto x = get<0>(coords) + xi_offset;
    const auto& y = get<1>(coords);
    const auto& z = get<2>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(x.size());
    get(get<ScalarTag>(vars)) = 0.8 * x + y - 0.4 * z + 0.5 * x * y * z;
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = z + 0.1 * square(x);
    get<2>(get<VectorTag<3>>(vars)) = y + square(x) * z;
    return vars;
  };
  const auto make_lower_eta_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto y = get<1>(coords) + eta_offset;
    const auto& z = get<2>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(x.size());
    get(get<ScalarTag>(vars)) = x + y - y * z + 0.2 * x * y * square(z);
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = -0.1 * y + z;
    get<2>(get<VectorTag<3>>(vars)) = -square(z);
    return vars;
  };
  const auto make_upper_eta_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto y = get<1>(coords) + eta_offset;
    const auto& z = get<2>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(x.size());
    get(get<ScalarTag>(vars)) = 1.5 * y - square(y) * z + x * y * square(z);
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = z + 0.4 * x * cube(z);
    get<2>(get<VectorTag<3>>(vars)) = y * z + square(y) + cube(z);
    return vars;
  };
  const auto make_lower_zeta_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double zeta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto z = get<2>(coords) + zeta_offset;
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(x.size());
    get(get<ScalarTag>(vars)) = 2.4 - 0.2 * z + 0.1 * x * square(y) * square(z);
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = 0.9 * z - 2. * x * z;
    get<2>(get<VectorTag<3>>(vars)) = y + cube(z);
    return vars;
  };
  const auto make_upper_zeta_vars =
      [](const tnsr::I<DataVector, 3, Frame::Logical>& coords,
         const double zeta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    const auto z = get<2>(coords) + zeta_offset;
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(x.size());
    get(get<ScalarTag>(vars)) = x - 0.4 * x * y * square(z);
    get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
    get<1>(get<VectorTag<3>>(vars)) = 1.3 * square(y) * square(z);
    get<2>(get<VectorTag<3>>(vars)) = -x * y * z + square(y);
    return vars;
  };

  const auto local_vars = make_center_vars(logical_coords);
  VariablesMap<3> neighbor_vars{};
  VariablesMap<3> neighbor_modified_vars{};

  const auto shift_vars_to_local_means = [&mesh, &local_vars ](
      const Variables<tmpl::list<ScalarTag, VectorTag<3>>>& input) noexcept {
    const auto& local_s = get<ScalarTag>(local_vars);
    const auto& local_v = get<VectorTag<3>>(local_vars);
    auto result = input;
    auto& s = get<ScalarTag>(result);
    auto& v = get<VectorTag<3>>(result);
    get(s) += mean_value(get(local_s), mesh) - mean_value(get(s), mesh);
    get<0>(v) +=
        mean_value(get<0>(local_v), mesh) - mean_value(get<0>(v), mesh);
    get<1>(v) +=
        mean_value(get<1>(local_v), mesh) - mean_value(get<1>(v), mesh);
    get<2>(v) +=
        mean_value(get<2>(local_v), mesh) - mean_value(get<2>(v), mesh);
    return result;
  };

  const auto lower_xi =
      std::make_pair(Direction<3>::lower_xi(), ElementId<3>(1));
  if (directions_of_external_boundaries.count(lower_xi.first) == 0) {
    neighbor_vars[lower_xi] = make_lower_xi_vars(logical_coords, -2.0);
    neighbor_modified_vars[lower_xi] =
        shift_vars_to_local_means(make_lower_xi_vars(logical_coords));
  }

  const auto upper_xi =
      std::make_pair(Direction<3>::upper_xi(), ElementId<3>(2));
  if (directions_of_external_boundaries.count(upper_xi.first) == 0) {
    neighbor_vars[upper_xi] = make_upper_xi_vars(logical_coords, 2.0);
    neighbor_modified_vars[upper_xi] =
        shift_vars_to_local_means(make_upper_xi_vars(logical_coords));
  }

  const auto lower_eta =
      std::make_pair(Direction<3>::lower_eta(), ElementId<3>(3));
  if (directions_of_external_boundaries.count(lower_eta.first) == 0) {
    neighbor_vars[lower_eta] = make_lower_eta_vars(logical_coords, -2.0);
    neighbor_modified_vars[lower_eta] =
        shift_vars_to_local_means(make_lower_eta_vars(logical_coords));
  }

  const auto upper_eta =
      std::make_pair(Direction<3>::upper_eta(), ElementId<3>(4));
  if (directions_of_external_boundaries.count(upper_eta.first) == 0) {
    neighbor_vars[upper_eta] = make_upper_eta_vars(logical_coords, 2.0);
    neighbor_modified_vars[upper_eta] =
        shift_vars_to_local_means(make_upper_eta_vars(logical_coords));
  }

  const auto lower_zeta =
      std::make_pair(Direction<3>::lower_zeta(), ElementId<3>(5));
  if (directions_of_external_boundaries.count(lower_zeta.first) == 0) {
    neighbor_vars[lower_zeta] = make_lower_zeta_vars(logical_coords, -2.0);
    neighbor_modified_vars[lower_zeta] =
        shift_vars_to_local_means(make_lower_zeta_vars(logical_coords));
  }

  const auto upper_zeta =
      std::make_pair(Direction<3>::upper_zeta(), ElementId<3>(6));
  if (directions_of_external_boundaries.count(upper_zeta.first) == 0) {
    neighbor_vars[upper_zeta] = make_upper_zeta_vars(logical_coords, 2.0);
    neighbor_modified_vars[upper_zeta] =
        shift_vars_to_local_means(make_upper_zeta_vars(logical_coords));
  }

  const auto neighbor_data = make_neighbor_data_from_neighbor_vars(
      element, mesh, element_size, neighbor_vars);

  // The 3D Simple WENO solution has slightly larger numerical error
  Approx custom_approx = Approx::custom().epsilon(1.e-11).scale(1.0);
  test_weno_work<3>(Limiters::WenoType::SimpleWeno,
                    Limiters::Weno_detail::DerivativeWeight::PowTwoEll, element,
                    mesh, element_size, local_vars, neighbor_data,
                    neighbor_modified_vars, custom_approx);
}

void test_hweno_1d(const std::unordered_set<Direction<1>>&
                       directions_of_external_boundaries = {}) noexcept {
  INFO("Test Hermite WENO limiter in 1D");
  CAPTURE(directions_of_external_boundaries);
  const auto element =
      TestHelpers::Limiters::make_element<1>(directions_of_external_boundaries);
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(1.2);

  // Functions to produce dummy data on each element
  const auto make_center_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(x.size());
    get(get<ScalarTag>(vars)) = 1.0 - 2.0 * x + square(x);
    get<0>(get<VectorTag<1>>(vars)) = 0.4 * x - 0.1 * square(x);
    return vars;
  };
  const auto make_lower_xi_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords,
         const double offset = 0.0) noexcept {
    const auto x = get<0>(coords) + offset;
    Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(x.size());
    get(get<ScalarTag>(vars)) = -2.0 - 10.0 * x - square(x);
    get<0>(get<VectorTag<1>>(vars)) = -0.1 + 0.3 * x - 0.1 * square(x);
    return vars;
  };
  const auto make_upper_xi_vars =
      [](const tnsr::I<DataVector, 1, Frame::Logical>& coords,
         const double offset = 0.0) noexcept {
    const auto x = get<0>(coords) + offset;
    Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(x.size());
    get(get<ScalarTag>(vars)) = -0.3 - x + 0.5 * square(x);
    get<0>(get<VectorTag<1>>(vars)) = 0.6 * x - 0.3 * square(x);
    return vars;
  };

  const auto local_vars = make_center_vars(logical_coords);
  VariablesMap<1> neighbor_vars{};
  VariablesMap<1> neighbor_modified_vars{};

  const auto lower_xi =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1));
  if (directions_of_external_boundaries.count(lower_xi.first) == 0) {
    neighbor_vars[lower_xi] = make_lower_xi_vars(logical_coords, -2.0);
  }

  const auto upper_xi =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2));
  if (directions_of_external_boundaries.count(upper_xi.first) == 0) {
    neighbor_vars[upper_xi] = make_upper_xi_vars(logical_coords, 2.0);
  }

  const auto neighbor_data = make_neighbor_data_from_neighbor_vars(
      element, mesh, element_size, neighbor_vars);

  if (directions_of_external_boundaries.count(lower_xi.first) == 0) {
    neighbor_modified_vars[lower_xi] =
        Variables<tmpl::list<ScalarTag, VectorTag<1>>>(
            mesh.number_of_grid_points());
    auto& mod_scalar = get<ScalarTag>(neighbor_modified_vars.at(lower_xi));
    Limiters::hweno_modified_neighbor_solution<ScalarTag>(
        make_not_null(&mod_scalar), get<ScalarTag>(local_vars), element, mesh,
        neighbor_data, lower_xi);
    auto& mod_vector = get<VectorTag<1>>(neighbor_modified_vars.at(lower_xi));
    Limiters::hweno_modified_neighbor_solution<VectorTag<1>>(
        make_not_null(&mod_vector), get<VectorTag<1>>(local_vars), element,
        mesh, neighbor_data, lower_xi);
  }

  if (directions_of_external_boundaries.count(upper_xi.first) == 0) {
    neighbor_modified_vars[upper_xi] =
        Variables<tmpl::list<ScalarTag, VectorTag<1>>>(
            mesh.number_of_grid_points());
    auto& mod_scalar = get<ScalarTag>(neighbor_modified_vars.at(upper_xi));
    Limiters::hweno_modified_neighbor_solution<ScalarTag>(
        make_not_null(&mod_scalar), get<ScalarTag>(local_vars), element, mesh,
        neighbor_data, upper_xi);
    auto& mod_vector = get<VectorTag<1>>(neighbor_modified_vars.at(upper_xi));
    Limiters::hweno_modified_neighbor_solution<VectorTag<1>>(
        make_not_null(&mod_vector), get<VectorTag<1>>(local_vars), element,
        mesh, neighbor_data, upper_xi);
  }

  test_weno_work<1>(Limiters::WenoType::Hweno,
                    Limiters::Weno_detail::DerivativeWeight::Unity, element,
                    mesh, element_size, local_vars, neighbor_data,
                    neighbor_modified_vars);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.Weno", "[Limiters][Unit]") {
  test_weno_option_parsing();
  test_weno_serialization();

  test_package_data_1d();
  test_package_data_2d();
  test_package_data_3d();

  // Here we test that the WENO limiter correctly does/doesn't activate,
  // assuming that the MinmodTci is being used. This is not a test of the TCI
  // itself -- that is already done elsewhere.
  test_weno_tci_1d();
  test_weno_tci_2d();
  test_weno_tci_3d();

  // Test simple WENO
  test_simple_weno_1d();
  test_simple_weno_2d();
  test_simple_weno_3d();

  // Test simple WENO with particular boundaries labeled as external
  test_simple_weno_1d({{Direction<1>::lower_xi()}});
  test_simple_weno_2d({{Direction<2>::lower_eta()}});
  test_simple_weno_2d({{Direction<2>::lower_xi(), Direction<2>::lower_eta(),
                        Direction<2>::upper_eta()}});
  test_simple_weno_3d({{Direction<3>::lower_zeta()}});
  test_simple_weno_3d({{Direction<3>::lower_xi(), Direction<3>::upper_xi(),
                        Direction<3>::lower_eta(), Direction<3>::lower_zeta(),
                        Direction<3>::lower_zeta()}});

  // Test HWENO
  // Note that the bulk of the HWENO is in the computation of the HWENO fits,
  // and these are independently tested. The goal here is to make sure we
  // correctly switch between the different WENO types.
  test_hweno_1d();
  test_hweno_1d({{Direction<1>::lower_xi()}});
}
