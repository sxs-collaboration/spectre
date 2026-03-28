// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <utility>
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
#include "Utilities/Gsl.hpp"

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
void insert_mortar_data(
    DataVector& mortar_data, const Mesh<2>& target_mortar_mesh,
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
  const auto creator = domain::creators::NonconformingSphericalShells(
      2.0, 3.0, 4.0, 0, 2, 5, 8, 11, nullptr, nullptr);
  const auto domain = creator.domain();
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
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.MortarInterpolator", "[Unit][Evolution]") {
  test_non_conforming_spheres();
}
