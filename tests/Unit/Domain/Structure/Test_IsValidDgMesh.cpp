// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IsValidDgMesh.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Topology.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Literals.hpp"

namespace domain {
namespace {
void test_1d() {
  const auto extents = std::array{3_st};
  const DirectionMap<1, Neighbors<1>> neighbors{};
  const ElementId<1> root_id{0_st};
  const ElementId<1> refined_id{0_st, std::array{SegmentId{1_st, 0_st}}};

  CHECK(is_valid_dg_mesh(
      Mesh<1>{extents, Spectral::bases::hypercube<1>,
              Spectral::quadratures::hypercube<1>},
      Element<1>{root_id, neighbors, topologies::hypercube<1>}));
  CHECK(is_valid_dg_mesh(
      Mesh<1>{extents, Spectral::bases::hypercube<1>,
              Spectral::quadratures::hypercube<1>},
      Element<1>{refined_id, neighbors, topologies::hypercube<1>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<1>{extents, Spectral::bases::hypercube<1>,
              Spectral::quadratures::hypercube<1>},
      Element<1>{root_id, neighbors, topologies::hypertorus<1>}));
  CHECK(is_valid_dg_mesh(
      Mesh<1>{extents, Spectral::bases::hypertorus<1>,
              Spectral::quadratures::hypertorus<1>},
      Element<1>{root_id, neighbors, topologies::hypertorus<1>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<1>{extents, Spectral::bases::hypertorus<1>,
              Spectral::quadratures::hypertorus<1>},
      Element<1>{refined_id, neighbors, topologies::hypertorus<1>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<1>{extents, Spectral::bases::hypertorus<1>,
              Spectral::quadratures::hypertorus<1>},
      Element<1>{root_id, neighbors, topologies::hypercube<1>}));
}

void test_2d() {
  const auto extents = std::array{3_st, 5_st};
  const DirectionMap<2, Neighbors<2>> neighbors{};
  const ElementId<2> root_id{0_st};
  const ElementId<2> xi_refined_id_lower{
      0_st, std::array{SegmentId{1_st, 0_st}, SegmentId{0_st, 0_st}}};
  const ElementId<2> xi_refined_id_upper{
      0_st, std::array{SegmentId{1_st, 1_st}, SegmentId{0_st, 0_st}}};
  const ElementId<2> eta_refined_id_lower{
      0_st, std::array{SegmentId{0_st, 0_st}, SegmentId{1_st, 0_st}}};

  CHECK(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::hypercube<2>,
              Spectral::quadratures::hypercube<2>},
      Element<2>{root_id, neighbors, topologies::hypercube<2>}));
  CHECK(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::hypercube<2>,
              Spectral::quadratures::hypercube<2>},
      Element<2>{xi_refined_id_lower, neighbors, topologies::hypercube<2>}));
  CHECK(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::hypercube<2>,
              Spectral::quadratures::hypercube<2>},
      Element<2>{eta_refined_id_lower, neighbors, topologies::hypercube<2>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::hypercube<2>,
              Spectral::quadratures::hypercube<2>},
      Element<2>{root_id, neighbors, topologies::hypertorus<2>}));
  CHECK(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::hypertorus<2>,
              Spectral::quadratures::hypertorus<2>},
      Element<2>{root_id, neighbors, topologies::hypertorus<2>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::hypertorus<2>,
              Spectral::quadratures::hypertorus<2>},
      Element<2>{xi_refined_id_lower, neighbors, topologies::hypertorus<2>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::hypertorus<2>,
              Spectral::quadratures::hypertorus<2>},
      Element<2>{eta_refined_id_lower, neighbors, topologies::hypertorus<2>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::hypertorus<2>,
              Spectral::quadratures::hypertorus<2>},
      Element<2>{root_id, neighbors, topologies::hypercube<2>}));
  CHECK(is_valid_dg_mesh(Mesh<2>{extents, Spectral::bases::annulus<>,
                                 Spectral::quadratures::annulus<>},
                         Element<2>{root_id, neighbors, topologies::annulus}));
  CHECK(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::annulus<>,
              Spectral::quadratures::annulus<>},
      Element<2>{xi_refined_id_lower, neighbors, topologies::annulus}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::annulus<>,
              Spectral::quadratures::annulus<>},
      Element<2>{eta_refined_id_lower, neighbors, topologies::annulus}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<2>{extents, Spectral::bases::annulus<>,
                               Spectral::quadratures::annulus<>},
                       Element<2>{root_id, neighbors, topologies::disk}));
  CHECK(is_valid_dg_mesh(
      Mesh<2>{extents, Spectral::bases::spherical_surface,
              Spectral::quadratures::spherical_surface},
      Element<2>{root_id, neighbors, topologies::spherical_surface}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<2>{extents, Spectral::bases::spherical_surface,
                               Spectral::quadratures::spherical_surface},
                       Element<2>{xi_refined_id_lower, neighbors,
                                  topologies::spherical_surface}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<2>{extents, Spectral::bases::spherical_surface,
                               Spectral::quadratures::spherical_surface},
                       Element<2>{eta_refined_id_lower, neighbors,
                                  topologies::spherical_surface}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<2>{extents, Spectral::bases::spherical_surface,
                               Spectral::quadratures::spherical_surface},
                       Element<2>{root_id, neighbors, topologies::disk}));
  const auto disk_extents = std::array{2_st, 5_st};
  CHECK(is_valid_dg_mesh(
      Mesh<2>{disk_extents, Spectral::bases::disk, Spectral::quadratures::disk},
      Element<2>{root_id, neighbors, topologies::disk}));
  CHECK(is_valid_dg_mesh(
      Mesh<2>{disk_extents, Spectral::bases::disk, Spectral::quadratures::disk},
      Element<2>{xi_refined_id_lower, neighbors, topologies::disk}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<2>{disk_extents, Spectral::bases::disk, Spectral::quadratures::disk},
      Element<2>{xi_refined_id_upper, neighbors, topologies::disk}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<2>{disk_extents, Spectral::bases::disk, Spectral::quadratures::disk},
      Element<2>{eta_refined_id_lower, neighbors, topologies::disk}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<2>{disk_extents, Spectral::bases::disk, Spectral::quadratures::disk},
      Element<2>{root_id, neighbors, topologies::annulus}));
}

void test_3d() {
  const auto extents = std::array{3_st, 5_st, 9_st};
  const DirectionMap<3, Neighbors<3>> neighbors{};
  const ElementId<3> root_id{0_st};
  const ElementId<3> xi_refined_id_lower{
      0_st, std::array{SegmentId{1_st, 0_st}, SegmentId{0_st, 0_st},
                       SegmentId{0_st, 0_st}}};
  const ElementId<3> xi_refined_id_upper{
      0_st, std::array{SegmentId{1_st, 1_st}, SegmentId{0_st, 0_st},
                       SegmentId{0_st, 0_st}}};
  const ElementId<3> eta_refined_id_lower{
      0_st, std::array{SegmentId{0_st, 0_st}, SegmentId{1_st, 0_st},
                       SegmentId{0_st, 0_st}}};
  const ElementId<3> zeta_refined_id_lower{
      0_st, std::array{SegmentId{0_st, 0_st}, SegmentId{0_st, 0_st},
                       SegmentId{1_st, 0_st}}};

  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypercube<3>,
              Spectral::quadratures::hypercube<3>},
      Element<3>{root_id, neighbors, topologies::hypercube<3>}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypercube<3>,
              Spectral::quadratures::hypercube<3>},
      Element<3>{xi_refined_id_lower, neighbors, topologies::hypercube<3>}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypercube<3>,
              Spectral::quadratures::hypercube<3>},
      Element<3>{eta_refined_id_lower, neighbors, topologies::hypercube<3>}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypercube<3>,
              Spectral::quadratures::hypercube<3>},
      Element<3>{zeta_refined_id_lower, neighbors, topologies::hypercube<3>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypercube<3>,
              Spectral::quadratures::hypercube<3>},
      Element<3>{root_id, neighbors, topologies::hypertorus<3>}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypertorus<3>,
              Spectral::quadratures::hypertorus<3>},
      Element<3>{root_id, neighbors, topologies::hypertorus<3>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypertorus<3>,
              Spectral::quadratures::hypertorus<3>},
      Element<3>{xi_refined_id_lower, neighbors, topologies::hypertorus<3>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypertorus<3>,
              Spectral::quadratures::hypertorus<3>},
      Element<3>{eta_refined_id_lower, neighbors, topologies::hypertorus<3>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypertorus<3>,
              Spectral::quadratures::hypertorus<3>},
      Element<3>{zeta_refined_id_lower, neighbors, topologies::hypertorus<3>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::hypertorus<3>,
              Spectral::quadratures::hypertorus<3>},
      Element<3>{root_id, neighbors, topologies::hypercube<3>}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::spherical_shell<>,
              Spectral::quadratures::spherical_shell<>},
      Element<3>{root_id, neighbors, topologies::spherical_shell}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::spherical_shell<>,
              Spectral::quadratures::spherical_shell<>},
      Element<3>{xi_refined_id_lower, neighbors, topologies::spherical_shell}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<3>{extents, Spectral::bases::spherical_shell<>,
                               Spectral::quadratures::spherical_shell<>},
                       Element<3>{eta_refined_id_lower, neighbors,
                                  topologies::spherical_shell}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<3>{extents, Spectral::bases::spherical_shell<>,
                               Spectral::quadratures::spherical_shell<>},
                       Element<3>{zeta_refined_id_lower, neighbors,
                                  topologies::spherical_shell}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::spherical_shell<>,
              Spectral::quadratures::spherical_shell<>},
      Element<3>{root_id, neighbors, topologies::full_sphere}));
  const auto cylindrical_extents = std::array{3_st, 9_st, 3_st};
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::cylindrical_shell<>,
              Spectral::quadratures::cylindrical_shell<>},
      Element<3>{root_id, neighbors, topologies::cylindrical_shell}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::cylindrical_shell<>,
              Spectral::quadratures::cylindrical_shell<>},
      Element<3>{xi_refined_id_lower, neighbors,
                 topologies::cylindrical_shell}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::cylindrical_shell<>,
              Spectral::quadratures::cylindrical_shell<>},
      Element<3>{eta_refined_id_lower, neighbors,
                 topologies::cylindrical_shell}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::cylindrical_shell<>,
              Spectral::quadratures::cylindrical_shell<>},
      Element<3>{zeta_refined_id_lower, neighbors,
                 topologies::cylindrical_shell}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::cylindrical_shell<>,
              Spectral::quadratures::cylindrical_shell<>},
      Element<3>{root_id, neighbors, topologies::full_cylinder}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::full_cylinder<>,
              Spectral::quadratures::full_cylinder<>},
      Element<3>{root_id, neighbors, topologies::full_cylinder}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::full_cylinder<>,
              Spectral::quadratures::full_cylinder<>},
      Element<3>{xi_refined_id_lower, neighbors, topologies::full_cylinder}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::full_cylinder<>,
              Spectral::quadratures::full_cylinder<>},
      Element<3>{xi_refined_id_upper, neighbors, topologies::full_cylinder}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::full_cylinder<>,
              Spectral::quadratures::full_cylinder<>},
      Element<3>{eta_refined_id_lower, neighbors, topologies::full_cylinder}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::full_cylinder<>,
              Spectral::quadratures::full_cylinder<>},
      Element<3>{zeta_refined_id_lower, neighbors, topologies::full_cylinder}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cylindrical_extents, Spectral::bases::full_cylinder<>,
              Spectral::quadratures::full_cylinder<>},
      Element<3>{root_id, neighbors, topologies::cylindrical_shell}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::full_sphere,
              Spectral::quadratures::full_sphere},
      Element<3>{root_id, neighbors, topologies::full_sphere}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::full_sphere,
              Spectral::quadratures::full_sphere},
      Element<3>{xi_refined_id_lower, neighbors, topologies::full_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::full_sphere,
              Spectral::quadratures::full_sphere},
      Element<3>{xi_refined_id_upper, neighbors, topologies::full_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::full_sphere,
              Spectral::quadratures::full_sphere},
      Element<3>{eta_refined_id_lower, neighbors, topologies::full_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::full_sphere,
              Spectral::quadratures::full_sphere},
      Element<3>{zeta_refined_id_lower, neighbors, topologies::full_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{extents, Spectral::bases::full_sphere,
              Spectral::quadratures::full_sphere},
      Element<3>{root_id, neighbors, topologies::spherical_shell}));
  const auto cartoon_sphere_extents = std::array{3_st, 1_st, 1_st};
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere_inner,
              Spectral::quadratures::cartoon_sphere_inner},
      Element<3>{root_id, neighbors, topologies::cartoon_sphere_inner}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere_inner,
              Spectral::quadratures::cartoon_sphere_inner},
      Element<3>{xi_refined_id_lower, neighbors,
                 topologies::cartoon_sphere_inner}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere_inner,
              Spectral::quadratures::cartoon_sphere_inner},
      Element<3>{xi_refined_id_upper, neighbors,
                 topologies::cartoon_sphere_inner}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere_inner,
              Spectral::quadratures::cartoon_sphere_inner},
      Element<3>{eta_refined_id_lower, neighbors,
                 topologies::cartoon_sphere_inner}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere_inner,
              Spectral::quadratures::cartoon_sphere_inner},
      Element<3>{zeta_refined_id_lower, neighbors,
                 topologies::cartoon_sphere_inner}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere_inner,
              Spectral::quadratures::cartoon_sphere_inner},
      Element<3>{root_id, neighbors, topologies::full_sphere}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere<>,
              Spectral::quadratures::cartoon_sphere<>},
      Element<3>{root_id, neighbors, topologies::cartoon_sphere}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere<>,
              Spectral::quadratures::cartoon_sphere<>},
      Element<3>{xi_refined_id_lower, neighbors, topologies::cartoon_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere<>,
              Spectral::quadratures::cartoon_sphere<>},
      Element<3>{eta_refined_id_lower, neighbors, topologies::cartoon_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere<>,
              Spectral::quadratures::cartoon_sphere<>},
      Element<3>{zeta_refined_id_lower, neighbors,
                 topologies::cartoon_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_sphere_extents, Spectral::bases::cartoon_sphere<>,
              Spectral::quadratures::cartoon_sphere<>},
      Element<3>{root_id, neighbors, topologies::spherical_shell}));
  const auto cartoon_cylinder_extents = std::array{3_st, 3_st, 1_st};
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cartoon_cylinder_extents, Spectral::bases::cartoon_cylinder<>,
              Spectral::quadratures::cartoon_cylinder<>},
      Element<3>{root_id, neighbors, topologies::cartoon_cylinder}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cartoon_cylinder_extents, Spectral::bases::cartoon_cylinder<>,
              Spectral::quadratures::cartoon_cylinder<>},
      Element<3>{xi_refined_id_lower, neighbors,
                 topologies::cartoon_cylinder}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cartoon_cylinder_extents, Spectral::bases::cartoon_cylinder<>,
              Spectral::quadratures::cartoon_cylinder<>},
      Element<3>{eta_refined_id_lower, neighbors,
                 topologies::cartoon_cylinder}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_cylinder_extents, Spectral::bases::cartoon_cylinder<>,
              Spectral::quadratures::cartoon_cylinder<>},
      Element<3>{zeta_refined_id_lower, neighbors,
                 topologies::cartoon_cylinder}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_cylinder_extents, Spectral::bases::cartoon_cylinder<>,
              Spectral::quadratures::cartoon_cylinder<>},
      Element<3>{root_id, neighbors, topologies::cylindrical_shell}));
  CHECK(is_valid_dg_mesh(
      Mesh<3>{cartoon_cylinder_extents,
              Spectral::bases::cartoon_cylinder_inner<>,
              Spectral::quadratures::cartoon_cylinder_inner<>},
      Element<3>{root_id, neighbors, topologies::cartoon_cylinder_inner}));
  CHECK(
      is_valid_dg_mesh(Mesh<3>{cartoon_cylinder_extents,
                               Spectral::bases::cartoon_cylinder_inner<>,
                               Spectral::quadratures::cartoon_cylinder_inner<>},
                       Element<3>{xi_refined_id_lower, neighbors,
                                  topologies::cartoon_cylinder_inner}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<3>{cartoon_cylinder_extents,
                               Spectral::bases::cartoon_cylinder_inner<>,
                               Spectral::quadratures::cartoon_cylinder_inner<>},
                       Element<3>{xi_refined_id_upper, neighbors,
                                  topologies::cartoon_cylinder_inner}));
  CHECK(
      is_valid_dg_mesh(Mesh<3>{cartoon_cylinder_extents,
                               Spectral::bases::cartoon_cylinder_inner<>,
                               Spectral::quadratures::cartoon_cylinder_inner<>},
                       Element<3>{eta_refined_id_lower, neighbors,
                                  topologies::cartoon_cylinder_inner}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<3>{cartoon_cylinder_extents,
                               Spectral::bases::cartoon_cylinder_inner<>,
                               Spectral::quadratures::cartoon_cylinder_inner<>},
                       Element<3>{zeta_refined_id_lower, neighbors,
                                  topologies::cartoon_cylinder_inner}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{cartoon_cylinder_extents,
              Spectral::bases::cartoon_cylinder_inner<>,
              Spectral::quadratures::cartoon_cylinder_inner<>},
      Element<3>{root_id, neighbors, topologies::full_cylinder}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Structure.IsValidDgMesh", "[Domain][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}
}  // namespace domain
