// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <unordered_map>
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
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/SimpleWenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "DataStructures/DataBox/Prefixes.hpp"
// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"
// IWYU pragma: no_forward_declare Limiters::Weno
// IWYU pragma: no_forward_declare Tags::Mean
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare intrp::RegularGrid

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
  const auto hweno_1d_tvb =
      TestHelpers::test_creation<Limiters::Weno<1, tmpl::list<ScalarTag>>>(
          "Type: Hweno\n"
          "TvbConstant: 1.0");
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
  CHECK(hweno_1d != hweno_1d_tvb);
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
  const Limiters::Weno<1, tmpl::list<ScalarTag>> expected_hweno_1d_tvb(
      Limiters::WenoType::Hweno, 0.001, 1.0);
  const Limiters::Weno<1, tmpl::list<ScalarTag>> expected_hweno_1d_disabled(
      Limiters::WenoType::Hweno, 0.001, 0.0, true);
  const Limiters::Weno<1, tmpl::list<ScalarTag>> expected_simple_weno_1d(
      Limiters::WenoType::SimpleWeno, 0.001);
  const Limiters::Weno<2, tmpl::list<ScalarTag>> expected_hweno_2d(
      Limiters::WenoType::Hweno, 0.001);
  const Limiters::Weno<3, tmpl::list<ScalarTag, VectorTag<3>>>
      expected_hweno_3d_larger_weight(Limiters::WenoType::Hweno, 0.01, 0.0,
                                      true);
  CHECK(hweno_1d == expected_hweno_1d);
  CHECK(hweno_1d_larger_weight == expected_hweno_1d_larger_weight);
  CHECK(hweno_1d_tvb == expected_hweno_1d_tvb);
  CHECK(hweno_1d_disabled == expected_hweno_1d_disabled);
  CHECK(simple_weno_1d == expected_simple_weno_1d);
  CHECK(hweno_2d == expected_hweno_2d);
  CHECK(hweno_3d_larger_weight == expected_hweno_3d_larger_weight);
}

void test_weno_serialization() noexcept {
  INFO("Test WENO serialization");
  const Limiters::Weno<1, tmpl::list<ScalarTag>> weno(Limiters::WenoType::Hweno,
                                                      0.01, 1.0, true);
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
Variables<tmpl::list<ScalarTag, VectorTag<VolumeDim>>> make_local_vars(
    const Mesh<VolumeDim>& mesh) noexcept;

template <>
Variables<tmpl::list<ScalarTag, VectorTag<1>>> make_local_vars(
    const Mesh<1>& mesh) noexcept {
  const auto logical_coords = logical_coordinates(mesh);
  const auto& x = get<0>(logical_coords);
  Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(
      mesh.number_of_grid_points());
  get(get<ScalarTag>(vars)) = 1.0 - 2.0 * x + square(x);
  get<0>(get<VectorTag<1>>(vars)) = 0.4 * x - 0.1 * square(x);
  return vars;
}

template <>
Variables<tmpl::list<ScalarTag, VectorTag<2>>> make_local_vars(
    const Mesh<2>& mesh) noexcept {
  const auto logical_coords = logical_coordinates(mesh);
  const auto& x = get<0>(logical_coords);
  const auto& y = get<1>(logical_coords);
  Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(
      mesh.number_of_grid_points());
  get(get<ScalarTag>(vars)) = x + y - 0.5 * square(x) + 0.5 * square(y);
  get<0>(get<VectorTag<2>>(vars)) = x + 2.5 * y;
  get<1>(get<VectorTag<2>>(vars)) =
      0.1 + 0.2 * x - 0.4 * y + 0.3 * square(x) * square(y);
  return vars;
}

template <>
Variables<tmpl::list<ScalarTag, VectorTag<3>>> make_local_vars(
    const Mesh<3>& mesh) noexcept {
  const auto logical_coords = logical_coordinates(mesh);
  const auto& x = get<0>(logical_coords);
  const auto& y = get<1>(logical_coords);
  const auto& z = get<2>(logical_coords);
  Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(
      mesh.number_of_grid_points());
  get(get<ScalarTag>(vars)) = x + y - 0.2 * z - y * z + x * y * square(z);
  get<0>(get<VectorTag<3>>(vars)) = 0.4 * x * y * z + square(z);
  get<1>(get<VectorTag<3>>(vars)) = 0.01 * x - 0.01 * y + z;
  get<2>(get<VectorTag<3>>(vars)) = x + 0.2 * y + cube(z);
  return vars;
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
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
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
std::unordered_map<
    std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
    typename Limiters::Weno<
        VolumeDim, tmpl::list<ScalarTag, VectorTag<VolumeDim>>>::PackagedData,
    boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
make_neighbor_data(const Mesh<VolumeDim>& mesh,
                   const Element<VolumeDim>& element,
                   const std::array<double, VolumeDim>& element_size) noexcept;

template <>
std::unordered_map<std::pair<Direction<1>, ElementId<1>>,
                   typename Limiters::Weno<
                       1, tmpl::list<ScalarTag, VectorTag<1>>>::PackagedData,
                   boost::hash<std::pair<Direction<1>, ElementId<1>>>>
make_neighbor_data(const Mesh<1>& mesh, const Element<1>& element,
                   const std::array<double, 1>& element_size) noexcept {
  const auto make_lower_xi_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = 16.4;
    get<0>(get<VectorTag<1>>(vars)) = 1.2;
    return vars;
  };
  const auto make_upper_xi_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<1>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = -9.5;
    get<0>(get<VectorTag<1>>(vars)) = 4.2;
    return vars;
  };

  VariablesMap<1> neighbor_vars{};
  const auto lower_xi =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1));
  const auto upper_xi =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2));
  neighbor_vars[lower_xi] = make_lower_xi_vars();
  neighbor_vars[upper_xi] = make_upper_xi_vars();

  return make_neighbor_data_from_neighbor_vars(mesh, element, element_size,
                                               neighbor_vars);
}

template <>
std::unordered_map<std::pair<Direction<2>, ElementId<2>>,
                   typename Limiters::Weno<
                       2, tmpl::list<ScalarTag, VectorTag<2>>>::PackagedData,
                   boost::hash<std::pair<Direction<2>, ElementId<2>>>>
make_neighbor_data(const Mesh<2>& mesh, const Element<2>& element,
                   const std::array<double, 2>& element_size) noexcept {
  const auto make_lower_xi_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = -3.2;
    get<0>(get<VectorTag<2>>(vars)) = -4.2;
    get<1>(get<VectorTag<2>>(vars)) = -1.7;
    return vars;
  };
  const auto make_upper_xi_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = 2.8;
    get<0>(get<VectorTag<2>>(vars)) = 3.1;
    get<1>(get<VectorTag<2>>(vars)) = 2.3;
    return vars;
  };
  const auto make_lower_eta_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = -2.7;
    get<0>(get<VectorTag<2>>(vars)) = 1.4;
    get<1>(get<VectorTag<2>>(vars)) = 4.5;
    return vars;
  };
  const auto make_upper_eta_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<2>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = 3.1;
    get<0>(get<VectorTag<2>>(vars)) = 2.4;
    get<1>(get<VectorTag<2>>(vars)) = -1.2;
    return vars;
  };

  VariablesMap<2> neighbor_vars{};
  const auto lower_xi =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>(1));
  const auto upper_xi =
      std::make_pair(Direction<2>::upper_xi(), ElementId<2>(2));
  const auto lower_eta =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>(3));
  const auto upper_eta =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>(4));
  neighbor_vars[lower_xi] = make_lower_xi_vars();
  neighbor_vars[upper_xi] = make_upper_xi_vars();
  neighbor_vars[lower_eta] = make_lower_eta_vars();
  neighbor_vars[upper_eta] = make_upper_eta_vars();

  return make_neighbor_data_from_neighbor_vars(mesh, element, element_size,
                                               neighbor_vars);
}

template <>
std::unordered_map<std::pair<Direction<3>, ElementId<3>>,
                   typename Limiters::Weno<
                       3, tmpl::list<ScalarTag, VectorTag<3>>>::PackagedData,
                   boost::hash<std::pair<Direction<3>, ElementId<3>>>>
make_neighbor_data(const Mesh<3>& mesh, const Element<3>& element,
                   const std::array<double, 3>& element_size) noexcept {
  const auto make_lower_xi_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = -3.2;
    get<0>(get<VectorTag<3>>(vars)) = 0.;
    get<1>(get<VectorTag<3>>(vars)) = -0.1;
    get<2>(get<VectorTag<3>>(vars)) = -2.1;
    return vars;
  };
  const auto make_upper_xi_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = 3.8;
    get<0>(get<VectorTag<3>>(vars)) = 0.;
    get<1>(get<VectorTag<3>>(vars)) = 0.1;
    get<2>(get<VectorTag<3>>(vars)) = 2.1;
    return vars;
  };
  const auto make_lower_eta_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = -2.5;
    get<0>(get<VectorTag<3>>(vars)) = 0.;
    get<1>(get<VectorTag<3>>(vars)) = 0.1;
    get<2>(get<VectorTag<3>>(vars)) = -0.4;
    return vars;
  };
  const auto make_upper_eta_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = 2.3;
    get<0>(get<VectorTag<3>>(vars)) = 0.;
    get<1>(get<VectorTag<3>>(vars)) = -0.1;
    get<2>(get<VectorTag<3>>(vars)) = 1.1;
    return vars;
  };
  const auto make_lower_zeta_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = 0.41;
    get<0>(get<VectorTag<3>>(vars)) = 0.;
    get<1>(get<VectorTag<3>>(vars)) = -2.3;
    ;
    get<2>(get<VectorTag<3>>(vars)) = -8.2;
    return vars;
  };
  const auto make_upper_zeta_vars = [&mesh]() noexcept {
    Variables<tmpl::list<ScalarTag, VectorTag<3>>> vars(
        mesh.number_of_grid_points());
    get(get<ScalarTag>(vars)) = -0.42;
    get<0>(get<VectorTag<3>>(vars)) = 0.;
    get<1>(get<VectorTag<3>>(vars)) = 2.3;
    get<2>(get<VectorTag<3>>(vars)) = 9.;
    return vars;
  };

  VariablesMap<3> neighbor_vars{};
  const auto lower_xi =
      std::make_pair(Direction<3>::lower_xi(), ElementId<3>(1));
  const auto upper_xi =
      std::make_pair(Direction<3>::upper_xi(), ElementId<3>(2));
  const auto lower_eta =
      std::make_pair(Direction<3>::lower_eta(), ElementId<3>(3));
  const auto upper_eta =
      std::make_pair(Direction<3>::upper_eta(), ElementId<3>(4));
  const auto lower_zeta =
      std::make_pair(Direction<3>::lower_zeta(), ElementId<3>(5));
  const auto upper_zeta =
      std::make_pair(Direction<3>::upper_zeta(), ElementId<3>(6));
  neighbor_vars[lower_xi] = make_lower_xi_vars();
  neighbor_vars[upper_xi] = make_upper_xi_vars();
  neighbor_vars[lower_eta] = make_lower_eta_vars();
  neighbor_vars[upper_eta] = make_upper_eta_vars();
  neighbor_vars[lower_zeta] = make_lower_zeta_vars();
  neighbor_vars[upper_zeta] = make_upper_zeta_vars();

  return make_neighbor_data_from_neighbor_vars(mesh, element, element_size,
                                               neighbor_vars);
}

template <size_t VolumeDim>
void test_simple_weno(const std::array<size_t, VolumeDim>& extents) noexcept {
  INFO("Test simple WENO limiter");
  CAPTURE(VolumeDim);
  const auto mesh = Mesh<VolumeDim>(extents, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<VolumeDim>();
  const auto element_size = make_array<VolumeDim>(1.2);

  // Make local + neighbor data where the vector x-component triggers limiting
  const auto local_vars = make_local_vars(mesh);
  const auto neighbor_data = make_neighbor_data(mesh, element, element_size);

  const double neighbor_linear_weight = 0.001;
  using Weno =
      Limiters::Weno<VolumeDim, tmpl::list<ScalarTag, VectorTag<VolumeDim>>>;
  const Weno simple_weno(Limiters::WenoType::SimpleWeno,
                         neighbor_linear_weight);

  auto scalar = get<ScalarTag>(local_vars);
  auto vector = get<VectorTag<VolumeDim>>(local_vars);
  const bool activated =
      simple_weno(make_not_null(&scalar), make_not_null(&vector), mesh, element,
                  element_size, neighbor_data);

  // Because simple WENO acts on each tensor component independently, only the
  // vector x-component should be modified by the limiter
  const auto& expected_scalar = get<ScalarTag>(local_vars);
  auto expected_vector = get<VectorTag<VolumeDim>>(local_vars);
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      intrp::RegularGrid<VolumeDim>,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      interpolator_buffer{};
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solution_buffer{};
  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    modified_neighbor_solution_buffer.insert(
        make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
  }
  Limiters::Weno_detail::simple_weno_impl<VectorTag<VolumeDim>>(
      make_not_null(&interpolator_buffer),
      make_not_null(&modified_neighbor_solution_buffer),
      make_not_null(&expected_vector), neighbor_linear_weight,
      0,  // the x-component
      mesh, element, neighbor_data);

  CHECK(activated);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_scalar, scalar, approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_vector, vector, approx);

  // Now call the limiter again, but this time use a non-zero TVB constant such
  // that the TCI says limiting isn't needed
  const double tvb_constant = 2.0;
  const Weno simple_weno_tvb(Limiters::WenoType::SimpleWeno,
                             neighbor_linear_weight, tvb_constant);

  scalar = get<ScalarTag>(local_vars);
  vector = get<VectorTag<VolumeDim>>(local_vars);
  const bool activated_tvb =
      simple_weno_tvb(make_not_null(&scalar), make_not_null(&vector), mesh,
                      element, element_size, neighbor_data);

  // expected_scalar is already set to local_vars
  expected_vector = get<VectorTag<VolumeDim>>(local_vars);

  CHECK_FALSE(activated_tvb);
  CHECK_ITERABLE_APPROX(expected_scalar, scalar);
  CHECK_ITERABLE_APPROX(expected_vector, vector);
}

template <size_t VolumeDim>
void test_hweno(const std::array<size_t, VolumeDim>& extents) noexcept {
  INFO("Test HWENO limiter");
  CAPTURE(VolumeDim);
  const auto mesh = Mesh<VolumeDim>(extents, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto);
  const auto element = TestHelpers::Limiters::make_element<VolumeDim>();
  const auto element_size = make_array<VolumeDim>(1.2);

  // Make local + neighbor data where the vector x-component triggers limiting
  const auto local_vars = make_local_vars(mesh);
  const auto neighbor_data = make_neighbor_data(mesh, element, element_size);

  const double neighbor_linear_weight = 0.001;
  using Weno =
      Limiters::Weno<VolumeDim, tmpl::list<ScalarTag, VectorTag<VolumeDim>>>;
  const Weno hweno(Limiters::WenoType::Hweno, neighbor_linear_weight);

  auto scalar = get<ScalarTag>(local_vars);
  auto vector = get<VectorTag<VolumeDim>>(local_vars);
  const bool activated = hweno(make_not_null(&scalar), make_not_null(&vector),
                               mesh, element, element_size, neighbor_data);

  // Because HWENO acts on the whole solution, the scalar and all vector
  // components should be modified by the limiter
  auto expected_scalar = get<ScalarTag>(local_vars);
  auto expected_vector = get<VectorTag<VolumeDim>>(local_vars);
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solution_buffer{};
  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    modified_neighbor_solution_buffer.insert(
        make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
  }
  Limiters::Weno_detail::hweno_impl<ScalarTag>(
      make_not_null(&modified_neighbor_solution_buffer),
      make_not_null(&expected_scalar), neighbor_linear_weight, mesh, element,
      neighbor_data);
  Limiters::Weno_detail::hweno_impl<VectorTag<VolumeDim>>(
      make_not_null(&modified_neighbor_solution_buffer),
      make_not_null(&expected_vector), neighbor_linear_weight, mesh, element,
      neighbor_data);

  CHECK(activated);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_scalar, scalar, approx);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_vector, vector, approx);

  // Now call the limiter again, but this time use a non-zero TVB constant such
  // that the TCI says limiting isn't needed
  const double tvb_constant = 2.0;
  const Weno hweno_tvb(Limiters::WenoType::Hweno, neighbor_linear_weight,
                       tvb_constant);

  scalar = get<ScalarTag>(local_vars);
  vector = get<VectorTag<VolumeDim>>(local_vars);
  const bool activated_tvb =
      hweno_tvb(make_not_null(&scalar), make_not_null(&vector), mesh, element,
                element_size, neighbor_data);

  expected_scalar = get<ScalarTag>(local_vars);
  expected_vector = get<VectorTag<VolumeDim>>(local_vars);

  CHECK_FALSE(activated_tvb);
  CHECK_ITERABLE_APPROX(expected_scalar, scalar);
  CHECK_ITERABLE_APPROX(expected_vector, vector);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.Weno", "[Limiters][Unit]") {
  test_weno_option_parsing();
  test_weno_serialization();

  test_package_data_1d();
  test_package_data_2d();
  test_package_data_3d();

  // The simple WENO reconstruction is tested in Test_SimpleWenoImpl.cpp.
  // Here we test that
  // - the TCI correctly acts component-by-component
  // - the limiter is indeed calling `simple_weno_impl`
  test_simple_weno<1>({{3}});
  test_simple_weno<2>({{3, 4}});
  test_simple_weno<3>({{3, 4, 5}});

  // The HWENO reconstruction is tested in Test_HwenoImpl.cpp.
  // Here we test that
  // - the TCI correctly triggers limiting on all tensors at once
  // - the limiter is indeed calling `hweno_impl`
  test_hweno<1>({{3}});
  test_hweno<2>({{3, 4}});
  test_hweno<3>({{3, 4, 5}});
}
