// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Disk.hpp"

#include <cmath>

#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
struct Logical;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {

template <typename TargetFrame>
Disk<TargetFrame>::Disk(
    typename InnerRadius::type inner_radius,
    typename OuterRadius::type outer_radius,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map) noexcept
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),         // NOLINT
      outer_radius_(std::move(outer_radius)),         // NOLINT
      initial_refinement_(                            // NOLINT
          std::move(initial_refinement)),             // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      use_equiangular_map_(use_equiangular_map) {}    // NOLINT

template <typename TargetFrame>
Domain<2, TargetFrame> Disk<TargetFrame>::create_domain() const noexcept {
  using Wedge2DMap = CoordinateMaps::Wedge2D;
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular2D =
      CoordinateMaps::ProductOf2Maps<Equiangular, Equiangular>;

  std::array<size_t, 4> block0_corners{{1, 5, 3, 7}},  //+x wedge
      block1_corners{{3, 7, 2, 6}},                    //+y wedge
      block2_corners{{2, 6, 0, 4}},                    //-x wedge
      block3_corners{{0, 4, 1, 5}},                    //-y wedge
      block4_corners{{0, 1, 2, 3}};                    // Center square

  std::vector<std::array<size_t, 4>> corners{block0_corners, block1_corners,
                                             block2_corners, block3_corners,
                                             block4_corners};

  auto coord_maps = make_vector_coordinate_map_base<Frame::Logical,
                                                    TargetFrame>(
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
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Equiangular2D{
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                        inner_radius_ / sqrt(2.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                        inner_radius_ / sqrt(2.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Affine2D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0))}));
  }
  return Domain<2, TargetFrame>{std::move(coord_maps), corners};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 2>> Disk<TargetFrame>::initial_extents() const
    noexcept {
  return {
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1]}}};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 2>>
Disk<TargetFrame>::initial_refinement_levels() const noexcept {
  return {5, make_array<2>(initial_refinement_)};
}

template class Disk<Frame::Grid>;
template class Disk<Frame::Inertial>;
}  // namespace creators
}  // namespace domain
