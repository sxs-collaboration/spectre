// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Disk.hpp"

#include <cmath>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame
/// \endcond

namespace domain::creators {
Disk::Disk(typename InnerRadius::type inner_radius,
           typename OuterRadius::type outer_radius,
           typename InitialRefinement::type initial_refinement,
           typename InitialGridPoints::type initial_number_of_grid_points,
           typename UseEquiangularMap::type use_equiangular_map,
           std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               boundary_condition,
           const Options::Context& context)
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),         // NOLINT
      outer_radius_(std::move(outer_radius)),         // NOLINT
      initial_refinement_(                            // NOLINT
          std::move(initial_refinement)),             // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      use_equiangular_map_(use_equiangular_map),      // NOLINT
      boundary_condition_(std::move(boundary_condition)) {
  using domain::BoundaryConditions::is_periodic;
  if (boundary_condition_ != nullptr and is_periodic(boundary_condition_)) {
    PARSE_ERROR(context, "Cannot have periodic boundary conditions on a disk.");
  }
}

Domain<2> Disk::create_domain() const noexcept {
  using Wedge2DMap = CoordinateMaps::Wedge2D;
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular2D =
      CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;

  std::array<size_t, 4> block0_corners{{1, 5, 3, 7}};  //+x wedge
  std::array<size_t, 4> block1_corners{{3, 7, 2, 6}};  //+y wedge
  std::array<size_t, 4> block2_corners{{2, 6, 0, 4}};  //-x wedge
  std::array<size_t, 4> block3_corners{{0, 4, 1, 5}};  //-y wedge
  std::array<size_t, 4> block4_corners{{0, 1, 2, 3}};  // Center square

  std::vector<std::array<size_t, 4>> corners{block0_corners, block1_corners,
                                             block2_corners, block3_corners,
                                             block4_corners};

  auto coord_maps = make_vector_coordinate_map_base<Frame::Logical,
                                                    Frame::Inertial>(
      Wedge2DMap{inner_radius_, outer_radius_, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                 use_equiangular_map_},
      Wedge2DMap{inner_radius_, outer_radius_, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                 use_equiangular_map_},
      Wedge2DMap{inner_radius_, outer_radius_, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                 use_equiangular_map_},
      Wedge2DMap{inner_radius_, outer_radius_, 0.0, 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                 use_equiangular_map_});

  if (use_equiangular_map_) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(Equiangular2D{
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                        inner_radius_ / sqrt(2.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                        inner_radius_ / sqrt(2.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Affine2D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0))}));
  }

  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (boundary_condition_ != nullptr) {
    for (size_t block_id = 0; block_id < 4; ++block_id) {
      DirectionMap<
          2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
          boundary_conditions{};
      boundary_conditions[Direction<2>::upper_xi()] =
          boundary_condition_->get_clone();
      boundary_conditions_all_blocks.push_back(std::move(boundary_conditions));
    }
    boundary_conditions_all_blocks.emplace_back();
  }

  return Domain<2>{std::move(coord_maps),
                   corners,
                   {},
                   std::move(boundary_conditions_all_blocks)};
}

std::vector<std::array<size_t, 2>> Disk::initial_extents() const noexcept {
  return {
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1]}}};
}

std::vector<std::array<size_t, 2>> Disk::initial_refinement_levels()
    const noexcept {
  return {5, make_array<2>(initial_refinement_)};
}
}  // namespace domain::creators
