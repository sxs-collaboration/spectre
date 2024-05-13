// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/SetInterpolators.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim, typename TargetFrame>
auto make_grid_map(const size_t id) {
  CAPTURE(id);
  REQUIRE((id == 0 or id == 1));
  using domain::make_coordinate_map_base;
  using domain::CoordinateMaps::Affine;
  if constexpr (Dim == 1) {
    return make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
        Affine(-1.0, 1.0, 2.0, id == 0 ? 5.0 : 0.0));
  } else if constexpr (Dim == 2) {
    using domain::CoordinateMaps::ProductOf2Maps;
    return make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
        ProductOf2Maps<Affine, Affine>(
            Affine(-1.0, 1.0, -1.0, id == 0 ? -0.8 : -1.9),
            Affine(-1.0, 1.0, -1.0, -0.8)));
  } else {
    using domain::CoordinateMaps::ProductOf3Maps;
    return make_coordinate_map_base<Frame::BlockLogical, TargetFrame>(
        ProductOf3Maps<Affine, Affine, Affine>(
            Affine(-1.0, 1.0, -1.0, id == 0 ? -0.8 : -1.9),
            Affine(-1.0, 1.0, -1.0, -0.8), Affine(-1.0, 1.0, 0.8, 1.0)));
  }
}

class DummyReconstructor {
 public:
  static size_t ghost_zone_size() { return 2; }
};

namespace Tags {
struct Reconstructor : db::SimpleTag,
                       evolution::dg::subcell::Tags::Reconstructor {
  using type = std::unique_ptr<DummyReconstructor>;
};
}  // namespace Tags

template <size_t Dim>
void test() {
  // Domain setup:
  //   | lower_xi | element |
  //
  // No neighbors in upper xi or in eta/zeta.

  const ::Mesh<Dim> dg_mesh{6, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const ::Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  const ElementId<Dim> element_id{0};
  const ElementId<Dim> lower_xi_element_id{1};
  DirectionMap<Dim, Neighbors<Dim>> element_neighbors{};
  const DirectionalId<Dim> lower_xi_id{Direction<Dim>::lower_xi(),
                                       lower_xi_element_id};
  OrientationMap<Dim> orientation{};
  if constexpr (Dim == 1) {
    orientation = OrientationMap<Dim>{{{Direction<Dim>::lower_xi()}}};
    element_neighbors.insert(
        std::pair{Direction<Dim>::lower_xi(),
                  Neighbors<Dim>{{lower_xi_id.id()}, orientation}});
  } else if constexpr (Dim == 2) {
    orientation = OrientationMap<Dim>{
        {{Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta()}}};
    element_neighbors.insert(
        std::pair{Direction<Dim>::lower_xi(),
                  Neighbors<Dim>{{lower_xi_id.id()}, orientation}});
  } else if constexpr (Dim == 3) {
    orientation = OrientationMap<Dim>{
        {{Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta(),
          Direction<Dim>::upper_zeta()}}};
    element_neighbors.insert(
        std::pair{Direction<Dim>::lower_xi(),
                  Neighbors<Dim>{{lower_xi_id.id()}, orientation}});
  }
  const Element<Dim> element{element_id, element_neighbors};

  DirectionMap<Dim, BlockNeighbor<Dim>> block0_neighbors{};
  block0_neighbors[Direction<Dim>::lower_xi()] =
      BlockNeighbor<Dim>{1, orientation};
  Block<Dim> block0{make_grid_map<Dim, Frame::Inertial>(0), 0,
                    block0_neighbors};
  block0.inject_time_dependent_map(
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{}));

  DirectionMap<Dim, BlockNeighbor<Dim>> block1_neighbors{};
  block1_neighbors[Direction<Dim>::lower_xi()] =
      BlockNeighbor<Dim>{0, orientation.inverse_map()};
  Block<Dim> block1{make_grid_map<Dim, Frame::Inertial>(1), 1,
                    block1_neighbors};
  block1.inject_time_dependent_map(
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{}));
  std::vector<Block<Dim>> blocks{2};
  blocks[0] = std::move(block0);
  blocks[1] = std::move(block1);

  const auto element_logical_coords = logical_coordinates(dg_mesh);
  tnsr::I<DataVector, Dim, Frame::BlockLogical> block_logical_coords{};
  for (size_t i = 0; i < Dim; ++i) {
    block_logical_coords[i] = element_logical_coords[i];
  }

  // NOLINTNEXTLINE(google-build-using-namespace)
  using namespace evolution::dg::subcell::Tags;
  auto box = db::create<
      db::AddSimpleTags<
          InterpolatorsFromFdToNeighborFd<Dim>,
          InterpolatorsFromDgToNeighborFd<Dim>,
          InterpolatorsFromNeighborDgToFd<Dim>,

          ::domain::Tags::Element<Dim>, ::domain::Tags::Domain<Dim>,
          domain::Tags::Mesh<Dim>, evolution::dg::subcell::Tags::Mesh<Dim>,
          ::domain::Tags::ElementMap<Dim, Frame::Grid>, Tags::Reconstructor>,
      db::AddComputeTags<
          domain::Tags::LogicalCoordinates<Dim>,
          domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<Dim, Frame::Grid>,
              domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
          evolution::dg::subcell::Tags::LogicalCoordinatesCompute<Dim>,
          domain::Tags::MappedCoordinates<
              ::domain::Tags::ElementMap<Dim, Frame::Grid>,
              evolution::dg::subcell::Tags::Coordinates<Dim,
                                                        Frame::ElementLogical>,
              evolution::dg::subcell::Tags::Coordinates>>>(
      typename InterpolatorsFromFdToNeighborFd<Dim>::type{},
      typename InterpolatorsFromDgToNeighborFd<Dim>::type{},
      typename InterpolatorsFromNeighborDgToFd<Dim>::type{},

      element, Domain<Dim>{std::move(blocks)}, dg_mesh, subcell_mesh,
      ElementMap{element_id, make_grid_map<Dim, Frame::Grid>(0)},
      std::make_unique<DummyReconstructor>());
  db::mutate_apply<evolution::dg::subcell::SetInterpolators<Dim>>(
      make_not_null(&box));

  // Check that the interpolators were set.
  REQUIRE(db::get<InterpolatorsFromDgToNeighborFd<Dim>>(box).size() == 1);
  REQUIRE(db::get<InterpolatorsFromDgToNeighborFd<Dim>>(box)
              .at(lower_xi_id)
              .has_value());
  REQUIRE(db::get<InterpolatorsFromFdToNeighborFd<Dim>>(box).size() == 1);
  REQUIRE(db::get<InterpolatorsFromFdToNeighborFd<Dim>>(box)
              .at(lower_xi_id)
              .has_value());
  REQUIRE(db::get<InterpolatorsFromNeighborDgToFd<Dim>>(box).size() == 1);
  REQUIRE(db::get<InterpolatorsFromNeighborDgToFd<Dim>>(box)
              .at(lower_xi_id)
              .has_value());

  // Interpolate to neighbor ghost zones.
  tnsr::I<DataVector, Dim, Frame::Grid> dg_to_ghost_zones{};
  tnsr::I<DataVector, Dim, Frame::Grid> fd_to_ghost_zones{};
  for (size_t i = 0; i < Dim; ++i) {
    dg_to_ghost_zones.get(i) =
        db::get<InterpolatorsFromDgToNeighborFd<Dim>>(box)
            .at(lower_xi_id)
            .value()
            .interpolate(
                db::get<domain::Tags::Coordinates<Dim, Frame::Grid>>(box).get(
                    i));
    fd_to_ghost_zones.get(i) =
        db::get<InterpolatorsFromFdToNeighborFd<Dim>>(box)
            .at(lower_xi_id)
            .value()
            .interpolate(db::get<evolution::dg::subcell::Tags::Coordinates<
                             Dim, Frame::Grid>>(box)
                             .get(i));
  }
  // Compute the lower-xi neighbor expected ghost zones.
  const auto lower_xi_logical_ghost_coords =
      evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
          subcell_mesh, DummyReconstructor::ghost_zone_size(),
          Direction<Dim>::lower_xi());
  const ElementMap lower_xi_element_map{lower_xi_element_id,
                                        make_grid_map<Dim, Frame::Grid>(1)};
  const auto lower_xi_neighbor_grid_ghost_coords =
      lower_xi_element_map(lower_xi_logical_ghost_coords);
  // Check interpolation worked correctly.
  CHECK_ITERABLE_APPROX(dg_to_ghost_zones, lower_xi_neighbor_grid_ghost_coords);
  CHECK_ITERABLE_APPROX(fd_to_ghost_zones, lower_xi_neighbor_grid_ghost_coords);

  // Check interpolating lower-xi DG data works to our FD ghost zones.
  const auto lower_xi_dg_grid_coords =
      lower_xi_element_map(element_logical_coords);
  tnsr::I<DataVector, Dim, Frame::Grid> lower_xi_dg_to_ghost_zones{};
  for (size_t i = 0; i < Dim; ++i) {
    lower_xi_dg_to_ghost_zones.get(i) =
        db::get<InterpolatorsFromNeighborDgToFd<Dim>>(box)
            .at(lower_xi_id)
            .value()
            .interpolate(lower_xi_dg_grid_coords.get(i));
  }
  const auto expected_lower_xi_dg_to_ghost_zones =
      db::get<domain::Tags::ElementMap<Dim, Frame::Grid>>(box)(
          lower_xi_logical_ghost_coords);
  CHECK_ITERABLE_APPROX(lower_xi_dg_to_ghost_zones,
                        expected_lower_xi_dg_to_ghost_zones);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.SetInterpolators",
                  "[Evolution][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
