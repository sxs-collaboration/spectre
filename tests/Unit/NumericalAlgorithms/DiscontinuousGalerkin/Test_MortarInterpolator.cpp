// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
DataVector vars_shell(const Mesh<2>& shell_mortar_mesh) {
  const YlmTestFunctions::ProductOfPolynomials f1(1, 2, 3);
  const YlmTestFunctions::ProductOfPolynomials f2(3, 1, 2);
  const auto shell_theta_phi = logical_coordinates(shell_mortar_mesh);
  const DataVector f1_shell = f1(shell_theta_phi);
  const DataVector f2_shell = f2(shell_theta_phi);
  const size_t npts = shell_mortar_mesh.number_of_grid_points();
  DataVector result{2 * npts};
  std::copy(f1_shell.begin(), f1_shell.end(), result.begin());
  std::copy(f2_shell.begin(), f2_shell.end(),
            result.begin() + static_cast<ptrdiff_t>(npts));
  return result;
}

DataVector vars_cubed_sphere(
    const Domain<3>& domain, const ElementId<3>& neighbor_id,
    const std::vector<std::array<size_t, 3>>& refinement_levels,
    const Mesh<2>& cubed_sphere_mortar_mesh) {
  const Element<3> cubed_sphere = domain::create_initial_element(
      neighbor_id, domain.blocks(), refinement_levels);
  const auto xi = interface_logical_coordinates(cubed_sphere_mortar_mesh,
                                                Direction<3>::upper_zeta());
  const ElementMap<3, Frame::Inertial> cubed_sphere_map{
      neighbor_id, domain.blocks()[neighbor_id.block_id()]};
  const auto x_inertial = cubed_sphere_map(xi);
  const auto& x = get<0>(x_inertial);
  const auto& y = get<1>(x_inertial);
  const auto& z = get<2>(x_inertial);
  const auto theta = atan2(hypot(x, y), z);
  const auto phi = atan2(y, x);
  const YlmTestFunctions::ProductOfPolynomials f1(1, 2, 3);
  const YlmTestFunctions::ProductOfPolynomials f2(3, 1, 2);
  const DataVector f1_cubed_sphere = f1(theta, phi);
  const DataVector f2_cubed_sphere = f2(theta, phi);
  const size_t npts = cubed_sphere_mortar_mesh.number_of_grid_points();
  DataVector result{2 * npts};
  std::copy(f1_cubed_sphere.begin(), f1_cubed_sphere.end(), result.begin());
  std::copy(f2_cubed_sphere.begin(), f2_cubed_sphere.end(),
            result.begin() + static_cast<ptrdiff_t>(npts));
  return result;
}

template <size_t Dim>
void insert_mortar_data(DataVector& mortar_data,
                        const Mesh<2>& target_mortar_mesh,
                        const DataVector& source_data,
                        const dg::MortarInterpolator<Dim>& interpolator) {
  const auto& offsets = interpolator.interpolated_neighbor_data_offsets();
  const DataVector subset_of_mortar_data =
      interpolator.interpolate_to_neighbor(source_data);
  for (size_t i = 0; i < offsets.size(); ++i) {
    mortar_data[offsets[i]] = subset_of_mortar_data[i];
    mortar_data[offsets[i] + target_mortar_mesh.number_of_grid_points()] =
        subset_of_mortar_data[i + offsets.size()];
  }
}

void test_non_conforming_spheres() {
  using Excision = domain::creators::NonconformingSphericalShells::Excision;
  const auto creator = domain::creators::NonconformingSphericalShells(
      2.0, 3.0, 4.0, Excision{}, std::array<size_t, 2>{2_st, 0_st}, 0_st,
      std::array<size_t, 2>{11_st, 5_st}, std::array<size_t, 2>{8_st, 5_st});
  const auto domain = creator.create_domain();
  const auto refinement_levels = creator.initial_refinement_levels();
  const ElementId<3> shell_id{6};
  const Element<3> shell = domain::create_initial_element(
      shell_id, domain.blocks(), refinement_levels);
  const auto& shell_neighbor_ids =
      shell.neighbors().at(Direction<3>::lower_xi());
  const Mesh<2> shell_mortar_mesh{
      std::array{8_st, 15_st},
      std::array{Spectral::Basis::SphericalHarmonic,
                 Spectral::Basis::SphericalHarmonic},
      std::array{Spectral::Quadrature::Gauss,
                 Spectral::Quadrature::Equiangular}};
  const Mesh<2> shell_mortar_mesh_2{
      std::array{9_st, 17_st},
      std::array{Spectral::Basis::SphericalHarmonic,
                 Spectral::Basis::SphericalHarmonic},
      std::array{Spectral::Quadrature::Gauss,
                 Spectral::Quadrature::Equiangular}};
  const Mesh<2> cubed_sphere_mortar_mesh{11_st, Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto};
  const Mesh<2> cubed_sphere_mortar_mesh_2{12_st, Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto};
  const DataVector v_shell = vars_shell(shell_mortar_mesh);
  const DataVector v_shell_2 = vars_shell(shell_mortar_mesh_2);
  DataVector interpolated_v_shell{2 * shell_mortar_mesh.number_of_grid_points(),
                                  std::numeric_limits<double>::quiet_NaN()};
  DataVector interpolated_v_shell_2{
      2 * shell_mortar_mesh_2.number_of_grid_points(),
      std::numeric_limits<double>::quiet_NaN()};
  DataVector interpolated_v_shell_3{
      2 * shell_mortar_mesh_2.number_of_grid_points(),
      std::numeric_limits<double>::quiet_NaN()};
  for (const auto& neighbor_id : shell_neighbor_ids) {
    const DataVector v_cubed_sphere = vars_cubed_sphere(
        domain, neighbor_id, refinement_levels, cubed_sphere_mortar_mesh);
    dg::MortarInterpolator<3> interpolator{
        neighbor_id, DirectionalId<3>{Direction<3>::upper_zeta(), shell_id},
        domain, cubed_sphere_mortar_mesh, shell_mortar_mesh};
    CHECK(interpolator.neighbor_mortar_mesh() == shell_mortar_mesh);
    const DataVector interpolated_v_cubed_sphere =
        interpolator.interpolate_to_host(v_shell);
    CHECK_ITERABLE_APPROX(interpolated_v_cubed_sphere, v_cubed_sphere);
    insert_mortar_data(interpolated_v_shell, shell_mortar_mesh, v_cubed_sphere,
                       interpolator);
    interpolator.reset_if_necessary(domain, cubed_sphere_mortar_mesh,
                                    shell_mortar_mesh_2);
    CHECK(interpolator.neighbor_mortar_mesh() == shell_mortar_mesh_2);
    const DataVector interpolated_v_cubed_sphere_2 =
        interpolator.interpolate_to_host(v_shell_2);
    CHECK_ITERABLE_APPROX(interpolated_v_cubed_sphere_2, v_cubed_sphere);
    insert_mortar_data(interpolated_v_shell_2, shell_mortar_mesh_2,
                       v_cubed_sphere, interpolator);
    interpolator.reset_if_necessary(domain, cubed_sphere_mortar_mesh_2,
                                    shell_mortar_mesh_2);
    CHECK(interpolator.neighbor_mortar_mesh() == shell_mortar_mesh_2);
    const DataVector interpolated_v_cubed_sphere_3 =
        interpolator.interpolate_to_host(v_shell_2);
    const DataVector v_cubed_sphere_2 = vars_cubed_sphere(
        domain, neighbor_id, refinement_levels, cubed_sphere_mortar_mesh_2);
    CHECK_ITERABLE_APPROX(interpolated_v_cubed_sphere_3, v_cubed_sphere_2);
    insert_mortar_data(interpolated_v_shell_3, shell_mortar_mesh_2,
                       v_cubed_sphere_2, interpolator);
    test_serialization(interpolator);
    CHECK(interpolator.neighbor_mortar_mesh() == shell_mortar_mesh_2);
  }
  Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(interpolated_v_shell, v_shell, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(interpolated_v_shell_2, v_shell_2,
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(interpolated_v_shell_3, v_shell_2,
                               custom_approx);
}

// Check interpolation across both the upper radial face of the object shells
// and the lower radial face of the wavezone shell.
void check_bco_pairing(const Domain<3>& domain,
                       const std::vector<std::array<size_t, 3>>& refinements,
                       const std::vector<std::array<size_t, 3>>& extents,
                       const size_t shell_block_id,
                       const Direction<3>& direction_shell_to_wedges,
                       const std::array<size_t, 2>& wedge_block_range,
                       const Direction<3>& direction_wedge_to_shell,
                       const std::array<double, 3>& center) {
  CAPTURE(shell_block_id);
  // Shell face mesh (angular dims of the SH mesh)
  const auto& shell_extents = extents[shell_block_id];
  const Mesh<3> shell_mesh{
      shell_extents,
      std::array{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
                 Spectral::Basis::SphericalHarmonic},
      std::array{Spectral::Quadrature::GaussLobatto,
                 Spectral::Quadrature::Gauss,
                 Spectral::Quadrature::Equiangular}};
  const Mesh<2> shell_face_mesh = shell_mesh.slice_away(0);
  const size_t npts_shell = shell_face_mesh.number_of_grid_points();
  // Analytic function of angle about the object's center
  const YlmTestFunctions::ProductOfPolynomials f(1, 2, 3);
  const DataVector f_shell = f(logical_coordinates(shell_face_mesh));

  std::vector<size_t> shell_point_contributions(npts_shell, 0);
  double max_err_to_host = 0.0;
  double max_err_to_neighbor = 0.0;

  for (size_t wedge_block = wedge_block_range[0];
       wedge_block < wedge_block_range[1]; ++wedge_block) {
    const auto& ref = refinements[wedge_block];
    const Mesh<3> wedge_mesh{extents[wedge_block], Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
    const Mesh<2> wedge_face_mesh =
        wedge_mesh.slice_away(direction_wedge_to_shell.dimension());
    // Loop over all wedge elements abutting the shell
    const size_t dim_n = direction_wedge_to_shell.dimension();
    const std::array<size_t, 2> transverse_dims =
        dim_n == 2
            ? std::array{0_st, 1_st}
            : (dim_n == 0 ? std::array{1_st, 2_st} : std::array{0_st, 2_st});
    const size_t normal_index = direction_wedge_to_shell.side() == Side::Lower
                                    ? 0
                                    : two_to_the(gsl::at(ref, dim_n)) - 1;
    for (size_t i = 0; i < two_to_the(gsl::at(ref, transverse_dims[0])); ++i) {
      for (size_t j = 0; j < two_to_the(gsl::at(ref, transverse_dims[1]));
           ++j) {
        std::array<SegmentId, 3> segments{};
        gsl::at(segments, transverse_dims[0]) =
            SegmentId{gsl::at(ref, transverse_dims[0]), i};
        gsl::at(segments, transverse_dims[1]) =
            SegmentId{gsl::at(ref, transverse_dims[1]), j};
        gsl::at(segments, dim_n) = SegmentId{gsl::at(ref, dim_n), normal_index};
        const ElementId<3> wedge_id{wedge_block, segments};
        CAPTURE(wedge_id);
        const Element<3> wedge_element = domain::create_initial_element(
            wedge_id, domain.blocks(), refinements);
        REQUIRE(wedge_element.neighbors().contains(direction_wedge_to_shell));
        const auto& shell_neighbors =
            wedge_element.neighbors().at(direction_wedge_to_shell);
        REQUIRE(shell_neighbors.size() == 1);
        const ElementId<3> shell_id = *shell_neighbors.begin();
        CAPTURE(shell_id);
        CHECK(shell_id.block_id() == shell_block_id);
        // Check the shell element is on the correct radial side
        {
          const auto shell_radial_segment = shell_id.segment_id(0);
          const size_t expected_index =
              direction_shell_to_wedges.side() == Side::Lower
                  ? 0
                  : two_to_the(shell_radial_segment.refinement_level()) - 1;
          CHECK(shell_radial_segment.index() == expected_index);
        }

        const dg::MortarInterpolator<3> interpolator{
            wedge_id, DirectionalId<3>{direction_wedge_to_shell, shell_id},
            domain, wedge_face_mesh, shell_face_mesh};

        // f at the wedge face points (in grid coordinates about the center)
        const auto xi = interface_logical_coordinates(wedge_face_mesh,
                                                      direction_wedge_to_shell);
        const ElementMap<3, Frame::Grid> wedge_map{
            wedge_id, domain.blocks()[wedge_block]};
        const auto x_grid = wedge_map(xi);
        const DataVector x = get<0>(x_grid) - center[0];
        const DataVector y = get<1>(x_grid) - center[1];
        const DataVector z = get<2>(x_grid) - center[2];
        const DataVector theta = atan2(hypot(x, y), z);
        DataVector phi = atan2(y, x);
        const DataVector f_wedge = f(theta, phi);

        // Shell -> wedge
        const DataVector f_wedge_interpolated =
            interpolator.interpolate_to_host(f_shell);
        CHECK(alg::all_of(f_wedge_interpolated, [](const double value) {
          return std::isfinite(value);
        }));
        for (size_t s = 0; s < f_wedge.size(); ++s) {
          max_err_to_host = std::max(
              max_err_to_host, std::abs(f_wedge_interpolated[s] - f_wedge[s]));
        }

        // Wedge -> shell
        const DataVector f_shell_subset =
            interpolator.interpolate_to_neighbor(f_wedge);
        CHECK(alg::all_of(f_shell_subset, [](const double value) {
          return std::isfinite(value);
        }));
        const auto& offsets = interpolator.interpolated_neighbor_data_offsets();
        REQUIRE(f_shell_subset.size() == offsets.size());
        for (size_t s = 0; s < offsets.size(); ++s) {
          ++shell_point_contributions[offsets[s]];
          max_err_to_neighbor =
              std::max(max_err_to_neighbor,
                       std::abs(f_shell_subset[s] - f_shell[offsets[s]]));
        }
      }
    }
  }
  CAPTURE(max_err_to_host);
  CAPTURE(max_err_to_neighbor);
  CHECK(max_err_to_host < 1.0e-3);
  CHECK(max_err_to_neighbor < 1.0e-3);
  // Every shell face point must be covered by at least one wedge element
  CHECK(alg::none_of(shell_point_contributions,
                     [](const size_t n) { return n == 0; }));
}

void test_bco_object_shells() {
  INFO("BCO with per-object SH shells and SH wavezone");
  using Object = domain::creators::BinaryCompactObject::Object;
  using GridPointsMap = std::unordered_map<
      std::string, std::variant<std::array<size_t, 3>, std::array<size_t, 2>>>;
  using RefinementMap =
      std::unordered_map<std::string,
                         std::variant<std::array<size_t, 3>, size_t>>;
  const domain::creators::BinaryCompactObject bco{
      Object{1.6, 2.7, 7.0, true, 1.8, 1.3, true},
      Object{1.9, 2.7, -1e-64, true, std::nullopt, 1.4, true},
      std::array<double, 2>{{0.0, 0.0}},
      50.0,
      500.0,
      1.0,
      RefinementMap{{"ObjectAShell", size_t{1}},
                    {"ObjectACube", std::array<size_t, 3>{2, 2, 1}},
                    {"ObjectBShell", size_t{1}},
                    {"ObjectBCube", std::array<size_t, 3>{2, 2, 1}},
                    {"Envelope", std::array<size_t, 3>{1, 1, 1}},
                    {"OuterShell0", size_t{3}}},
      GridPointsMap{{"ObjectAShell", std::array<size_t, 2>{10, 11}},
                    {"ObjectACube", std::array<size_t, 3>{10, 10, 9}},
                    {"ObjectBShell", std::array<size_t, 2>{10, 11}},
                    {"ObjectBCube", std::array<size_t, 3>{10, 10, 9}},
                    {"Envelope", std::array<size_t, 3>{12, 12, 15}},
                    {"OuterShell0", std::array<size_t, 2>{16, 11}}},
      true,
      domain::CoordinateMaps::Distribution::Linear,
      std::vector<double>{},
      domain::CoordinateMaps::Distribution::Linear,
      120.0,
      true,
      false,
      std::nullopt,
      nullptr};
  const auto domain = bco.create_domain();
  const auto refinements = bco.initial_refinement_levels();
  const auto extents = bco.initial_extents();

  {
    INFO("ObjectA pairing (shell block 0, cubes 1-6)");
    check_bco_pairing(domain, refinements, extents, 0, Direction<3>::upper_xi(),
                      {{1, 7}}, Direction<3>::lower_zeta(), {{7.0, 0.0, 0.0}});
  }
  {
    INFO("ObjectB pairing (shell block 7, cubes 8-13)");
    check_bco_pairing(domain, refinements, extents, 7, Direction<3>::upper_xi(),
                      {{8, 14}}, Direction<3>::lower_zeta(),
                      {{-1e-64, 0.0, 0.0}});
  }
  {
    INFO("Wavezone pairing (shell block 24, envelope 14-24)");
    check_bco_pairing(domain, refinements, extents, 24,
                      Direction<3>::lower_xi(), {{14, 24}},
                      Direction<3>::upper_zeta(), {{0.0, 0.0, 0.0}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.MortarInterpolator", "[Unit][Evolution]") {
  test_non_conforming_spheres();
  test_bco_object_shells();
}
