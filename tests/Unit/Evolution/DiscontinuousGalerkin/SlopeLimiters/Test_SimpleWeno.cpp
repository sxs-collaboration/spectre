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
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/SimpleWeno.hpp"
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

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.SimpleWeno.Serialization",
                  "[SlopeLimiters][Unit]") {
  const SlopeLimiters::SimpleWeno<1, tmpl::list<scalar, vector<1>>> sweno(
      0.001);
  test_serialization(sweno);

  const SlopeLimiters::SimpleWeno<1, tmpl::list<scalar, vector<1>>> sweno2(
      0.001, true);
  CHECK(sweno != sweno2);
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

// template <>
// Element<3> make_element() noexcept {
//  return Element<3>{
//      ElementId<3>{0},
//      Element<3>::Neighbors_t{
//          {Direction<3>::lower_xi(), make_neighbor_with_id<3>(1)},
//          {Direction<3>::upper_xi(), make_neighbor_with_id<3>(2)},
//          {Direction<3>::lower_eta(), make_neighbor_with_id<3>(3)},
//          {Direction<3>::upper_eta(), make_neighbor_with_id<3>(4)},
//          {Direction<3>::lower_zeta(), make_neighbor_with_id<3>(5)},
//          {Direction<3>::upper_zeta(), make_neighbor_with_id<3>(6)}}};
//}
}  // namespace

void test_simple_weno_1d(const double rhs_constant,
                         const bool expected_activation) noexcept {
  using SimpleWenoLimiter = SlopeLimiters::SimpleWeno<1, tmpl::list<scalar>>;

  const auto element = make_element<1>();
  const auto mesh =
      Mesh<1>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array<1>(1.2);

  // functions
  // u = 1 - 2 x + x^2
  const auto make_center_vars = [](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    Variables<tmpl::list<scalar>> vars(x.size());
    get(get<scalar>(vars)) = 1.0 - 2.0 * x + square(x);
    return vars;
  };
  // u = -2 - 10 x - x^2
  const auto make_lower_xi_vars = [](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords,
      const double offset = 0.0) noexcept {
    const auto& x = get<0>(coords) - offset;
    Variables<tmpl::list<scalar>> vars(x.size());
    get(get<scalar>(vars)) = -2.0 - 10.0 * x - square(x);
    return vars;
  };
  // u = rhs_constant - x + 0.5 x^2
  const auto make_upper_xi_vars = [&rhs_constant](
      const tnsr::I<DataVector, 1, Frame::Logical>& coords,
      const double offset = 0.0) noexcept {
    const auto& x = get<0>(coords) - offset;
    Variables<tmpl::list<scalar>> vars(x.size());
    get(get<scalar>(vars)) = rhs_constant - x + 0.5 * square(x);
    return vars;
  };

  // make data on main element
  const auto vars = make_center_vars(logical_coords);
  // make data on lower_xi neighbor
  const auto lower_xi_mesh =
      Mesh<1>(4, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto lower_xi_logical_coords = logical_coordinates(lower_xi_mesh);
  // offset coords, so evaluation gives values on neighbor's grid points
  const auto lower_xi_vars = make_lower_xi_vars(lower_xi_logical_coords, 2.0);
  // make data on upper_xi neighbor
  // offset coords, so evaluation gives values on neighbor's grid points
  const auto upper_xi_vars = make_upper_xi_vars(logical_coords, -2.0);

  // fill data into neighbor structure
  std::unordered_map<std::pair<Direction<1>, ElementId<1>>,
                     typename SimpleWenoLimiter::PackagedData,
                     boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data{};
  const auto lower_xi =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>(1));
  const auto upper_xi =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>(2));
  neighbor_data[lower_xi].volume_data = lower_xi_vars;
  neighbor_data[lower_xi].mesh = lower_xi_mesh;
  neighbor_data[upper_xi].volume_data = upper_xi_vars;
  neighbor_data[upper_xi].mesh = mesh;

  const SimpleWenoLimiter sweno(0.001);
  auto s = get<scalar>(vars);

  const bool activated = sweno(make_not_null(&s), element, mesh, logical_coords,
                               element_size, neighbor_data);

  if (expected_activation) {
    CHECK(activated);

    // expect smoothness indicators:
    // center_beta = integral (-2 + 2x)^2 dx
    // lower_xi_beta = integral (-10 -2x)^2 dx
    // upper_xi_beta = integral (-1 + x)^2 dx
    const double center_beta = 32.0 / 3.0;
    const double lower_xi_beta = 608.0 / 3.0;
    const double upper_xi_beta = 8.0 / 3.0;
    // unnormalized weights
    const double gamma = 0.001;
    const double center_gamma = 1.0 - 2 * gamma;
    double center_omega = center_gamma / square(1.e-6 + center_beta);
    double lower_xi_omega = gamma / square(1.e-6 + lower_xi_beta);
    double upper_xi_omega = gamma / square(1.e-6 + upper_xi_beta);
    const double norm = center_omega + lower_xi_omega + upper_xi_omega;
    center_omega /= norm;
    lower_xi_omega /= norm;
    upper_xi_omega /= norm;

    // expected solution:
    const auto lower_xi_vars_on_center = make_lower_xi_vars(logical_coords);
    const auto upper_xi_vars_on_center = make_upper_xi_vars(logical_coords);
    const double center_mean = mean_value(get(get<scalar>(vars)), mesh);
    const double lower_xi_mean =
        mean_value(get(get<scalar>(lower_xi_vars_on_center)), mesh);
    const double upper_xi_mean =
        mean_value(get(get<scalar>(upper_xi_vars_on_center)), mesh);

    DataVector expected_scalar =
        center_mean + center_omega * (get(get<scalar>(vars)) - center_mean) +
        lower_xi_omega *
            (get(get<scalar>(lower_xi_vars_on_center)) - lower_xi_mean) +
        upper_xi_omega *
            (get(get<scalar>(upper_xi_vars_on_center)) - upper_xi_mean);

    CHECK_ITERABLE_APPROX(expected_scalar, get(s));
  } else {
    CHECK_FALSE(activated);
  }
}

void test_simple_weno_2d(const double xi_rhs_constant,
                         const double eta_rhs_constant,
                         const bool expected_activation) noexcept {
  using SimpleWenoLimiter = SlopeLimiters::SimpleWeno<2, tmpl::list<scalar>>;

  const auto element = make_element<2>();
  const auto mesh =
      Mesh<2>(3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto logical_coords = logical_coordinates(mesh);
  const auto element_size = make_array(1.2, 1.2);

  // functions
  // u = x + y - 0.5 x^2 + 0.5 y^2
  const auto make_center_vars = [](
      const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords);
    Variables<tmpl::list<scalar>> vars(x.size());
    get(get<scalar>(vars)) = x + y - 0.5 * square(x) + 0.5 * square(y);
    return vars;
  };
  // u = 2 x + 1.2 y
  const auto make_lower_xi_vars = [](
      const tnsr::I<DataVector, 2, Frame::Logical>& coords,
      const double xi_offset = 0.0) noexcept {
    const auto& x = get<0>(coords) - xi_offset;
    const auto& y = get<1>(coords);
    Variables<tmpl::list<scalar>> vars(x.size());
    get(get<scalar>(vars)) = 2.0 * x + 1.2 * y;
    return vars;
  };
  // u = xi_rhs_constant + x + y - 0.25 x^2 - y^2
  const auto make_upper_xi_vars = [&xi_rhs_constant](
      const tnsr::I<DataVector, 2, Frame::Logical>& coords,
      const double xi_offset = 0.0) noexcept {
    const auto& x = get<0>(coords) - xi_offset;
    const auto& y = get<1>(coords);
    Variables<tmpl::list<scalar>> vars(x.size());
    get(get<scalar>(vars)) =
        xi_rhs_constant + x + y - 0.25 * square(x) - square(y);
    return vars;
  };
  // u = -1 + 0.5 x + y
  const auto make_lower_eta_vars = [](
      const tnsr::I<DataVector, 2, Frame::Logical>& coords,
      const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords) - eta_offset;
    Variables<tmpl::list<scalar>> vars(x.size());
    get(get<scalar>(vars)) = -1 + 0.5 * x + y;
    return vars;
  };
  // u = eta_rhs_constant + x + 2 y + 0.5 x^2 + 0.5 y^2
  const auto make_upper_eta_vars = [&eta_rhs_constant](
      const tnsr::I<DataVector, 2, Frame::Logical>& coords,
      const double eta_offset = 0.0) noexcept {
    const auto& x = get<0>(coords);
    const auto& y = get<1>(coords) - eta_offset;
    Variables<tmpl::list<scalar>> vars(x.size());
    get(get<scalar>(vars)) =
        eta_rhs_constant + x + 2.0 * y + 0.5 * square(x) + 0.5 * square(y);
    return vars;
  };

  // make data on main element
  const auto vars = make_center_vars(logical_coords);
  // make data on lower_xi neighbor
  const auto lower_xi_mesh =
      Mesh<2>(4, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
  const auto lower_xi_logical_coords = logical_coordinates(lower_xi_mesh);
  const auto lower_xi_vars = make_lower_xi_vars(lower_xi_logical_coords, 2.0);
  // make data on upper_xi neighbor
  const auto upper_xi_vars = make_upper_xi_vars(logical_coords, -2.0);
  // make data on lower_eta neighbor
  const auto lower_eta_vars = make_lower_eta_vars(logical_coords, 2.0);
  // make data on upper_eta neighbor
  const auto upper_eta_vars = make_upper_eta_vars(logical_coords, -2.0);

  // fill data into neighbor structure
  std::unordered_map<std::pair<Direction<2>, ElementId<2>>,
                     typename SimpleWenoLimiter::PackagedData,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  const auto lower_xi =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>(1));
  const auto upper_xi =
      std::make_pair(Direction<2>::upper_xi(), ElementId<2>(2));
  const auto lower_eta =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>(3));
  const auto upper_eta =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>(4));
  neighbor_data[lower_xi].volume_data = lower_xi_vars;
  neighbor_data[lower_xi].mesh = lower_xi_mesh;
  neighbor_data[upper_xi].volume_data = upper_xi_vars;
  neighbor_data[upper_xi].mesh = mesh;
  neighbor_data[lower_eta].volume_data = lower_eta_vars;
  neighbor_data[lower_eta].mesh = mesh;
  neighbor_data[upper_eta].volume_data = upper_eta_vars;
  neighbor_data[upper_eta].mesh = mesh;

  const SimpleWenoLimiter sweno(0.001);
  auto s = get<scalar>(vars);

  const bool activated = sweno(make_not_null(&s), element, mesh, logical_coords,
                               element_size, neighbor_data);

  if (expected_activation) {
    CHECK(activated);

    // expect smoothness indicators:
    // center_beta = integral (1 - x)^2 + (1 - y)^2 dx dy
    // lower_xi_beta = integral (2)^2 + (1.2)^2 dx dy
    // upper_xi_beta = integral (1 - 0.5 x)^2 + (1 - 2 y)^2 dx dy
    // lower_eta_beta = integral (0.5)^2 + (1)^2 dx dy
    // upper_eta_beta = integral (1 + x)^2 + (2 + y)^2 dx dy
    const double center_beta = 32.0 / 3.0;
    const double lower_xi_beta = 544.0 / 25.0;
    const double upper_xi_beta = 41.0 / 3.0;
    const double lower_eta_beta = 5.0;
    const double upper_eta_beta = 68.0 / 3.0;
    // unnormalized weights
    const double gamma = 0.001;
    const double center_gamma = 1.0 - 4 * gamma;
    double center_omega = center_gamma / square(1.e-6 + center_beta);
    double lower_xi_omega = gamma / square(1.e-6 + lower_xi_beta);
    double upper_xi_omega = gamma / square(1.e-6 + upper_xi_beta);
    double lower_eta_omega = gamma / square(1.e-6 + lower_eta_beta);
    double upper_eta_omega = gamma / square(1.e-6 + upper_eta_beta);
    const double norm = center_omega + lower_xi_omega + upper_xi_omega +
                        lower_eta_omega + upper_eta_omega;
    center_omega /= norm;
    lower_xi_omega /= norm;
    upper_xi_omega /= norm;
    lower_eta_omega /= norm;
    upper_eta_omega /= norm;

    // expected solution:
    const auto lower_xi_vars_on_center = make_lower_xi_vars(logical_coords);
    const auto upper_xi_vars_on_center = make_upper_xi_vars(logical_coords);
    const auto lower_eta_vars_on_center = make_lower_eta_vars(logical_coords);
    const auto upper_eta_vars_on_center = make_upper_eta_vars(logical_coords);
    const double center_mean = mean_value(get(get<scalar>(vars)), mesh);
    const double lower_xi_mean =
        mean_value(get(get<scalar>(lower_xi_vars_on_center)), mesh);
    const double upper_xi_mean =
        mean_value(get(get<scalar>(upper_xi_vars_on_center)), mesh);
    const double lower_eta_mean =
        mean_value(get(get<scalar>(lower_eta_vars_on_center)), mesh);
    const double upper_eta_mean =
        mean_value(get(get<scalar>(upper_eta_vars_on_center)), mesh);

    DataVector expected_scalar =
        center_mean + center_omega * (get(get<scalar>(vars)) - center_mean) +
        lower_xi_omega *
            (get(get<scalar>(lower_xi_vars_on_center)) - lower_xi_mean) +
        upper_xi_omega *
            (get(get<scalar>(upper_xi_vars_on_center)) - upper_xi_mean) +
        lower_eta_omega *
            (get(get<scalar>(lower_eta_vars_on_center)) - lower_eta_mean) +
        upper_eta_omega *
            (get(get<scalar>(upper_eta_vars_on_center)) - upper_eta_mean);

    CHECK_ITERABLE_APPROX(expected_scalar, get(s));
  } else {
    CHECK_FALSE(activated);
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.DG.SlopeLimiters.SimpleWeno.1d_detailed_action_test",
    "[SlopeLimiters][Unit]") {
  // SECTION("1D SimpleWeno: limiter activates") {
  //  test_simple_weno_1d(0.2, true);
  //  test_simple_weno_1d(0.33, true);
  //}
  // SECTION("1D SimpleWeno: troubled cell indicator avoids limiting") {
  //  test_simple_weno_1d(-1.0, false);
  //}
  SECTION("2D SimpleWeno: limiter activates") {
    test_simple_weno_2d(0.0, 0.0, true);
    test_simple_weno_2d(1.0, -6.0, true);
    test_simple_weno_2d(0.0, -6.0, true);
  }
  SECTION("2D SimpleWeno: troubled cell indicator avoids limiting") {
    test_simple_weno_2d(1.0, 0.0, false);
  }
}
