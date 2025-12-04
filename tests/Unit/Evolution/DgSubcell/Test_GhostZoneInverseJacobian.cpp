// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostZoneInverseJacobian.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/GhostZoneInverseJacobian.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace {

template <size_t Dim>
using CoordinateMap =
    tmpl::conditional_t<Dim == 3, domain::CoordinateMaps::Frustum,
                        domain::CoordinateMaps::Identity<Dim>>;

class DummyReconstructor {
 public:
  static size_t ghost_zone_size() { return 3; }
};

namespace Tags {
struct Reconstructor : db::SimpleTag {
  using type = std::unique_ptr<DummyReconstructor>;
};
}  // namespace Tags

template <size_t Dim>
void test() {
  // Since the mutator relies on already tested functions to compute each of the
  // quantities stored in the tag, we just test to make sure that they were
  // properly stored in the `GhostZoneInverseJacobian` tag upon its mutation
  CAPTURE(Dim);

  // Assemble an Element and its ElementMap
  const Mesh<Dim> dg_mesh{3, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using neighbor_tags = tmpl::list<
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>,
      evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>>;

  const ElementId<Dim> element_id{0};
  const Element<Dim> element{element_id, {}};

  CoordinateMap<Dim> coordinate_map;
  if constexpr (Dim == 3) {
    const std::array<std::array<double, 2>, 4> face_vertices{
        {{{-5., -5.}}, {{5., 5.}}, {{-3., -3.}}, {{3., 3.}}}};
    coordinate_map = domain::CoordinateMaps::Frustum(
        face_vertices, -4., 4., OrientationMap<3>::create_aligned());
  } else {
    coordinate_map = domain::CoordinateMaps::Identity<Dim>();
  }
  auto element_map = ElementMap<Dim, Frame::Grid>(
      element_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          coordinate_map));

  // Mutate the tag using the mutator
  auto box = db::create<db::AddSimpleTags<
      evolution::dg::subcell::Tags::GhostZoneInverseJacobian<Dim>,
      ::domain::Tags::Element<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
      ::domain::Tags::ElementMap<Dim, Frame::Grid>, Tags::Reconstructor>>(
      typename evolution::dg::subcell::Tags::GhostZoneInverseJacobian<
          Dim>::type{},
      element, subcell_mesh, std::move(element_map),
      std::make_unique<DummyReconstructor>());
  db::mutate_apply<evolution::dg::subcell::GhostZoneInverseJacobian<
      Dim, Tags::Reconstructor>>(make_not_null(&box));

  // Compare to computed outcome direction-wise
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto logical_coords =
        evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
            subcell_mesh, DummyReconstructor::ghost_zone_size(), direction);
    const auto grid_coords =
        db::get<::domain::Tags::ElementMap<Dim, Frame::Grid>>(box)(
            logical_coords);
    const auto inv_jacobian =
        db::get<::domain::Tags::ElementMap<Dim, Frame::Grid>>(box).inv_jacobian(
            logical_coords);

    const Variables<neighbor_tags> ghost_zone_inverse_jacobian =
        db::get<evolution::dg::subcell::Tags::GhostZoneInverseJacobian<Dim>>(
            box)
            .at(direction);
    const auto test_grid_coords =
        get<evolution::dg::subcell::Tags::Coordinates<Dim, Frame::Grid>>(
            ghost_zone_inverse_jacobian);
    const auto test_inv_jacobian = get<
        evolution::dg::subcell::fd::Tags::InverseJacobianLogicalToGrid<Dim>>(
        ghost_zone_inverse_jacobian);

    CHECK_ITERABLE_APPROX(grid_coords, test_grid_coords);
    CHECK_ITERABLE_APPROX(inv_jacobian, test_inv_jacobian);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.FD.GhostZoneInverseJacobian",
                  "[Parallel][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
