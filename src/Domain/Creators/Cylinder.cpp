// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Cylinder.hpp"

#include <cmath>

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Grid;
struct Inertial;
struct Logical;
}  // namespace Frame

namespace domain {
namespace creators {

template <typename TargetFrame>
Cylinder<TargetFrame>::Cylinder(
    typename InnerRadius::type inner_radius,
    typename OuterRadius::type outer_radius,
    typename LowerBound::type lower_bound,
    typename UpperBound::type upper_bound,
    typename IsPeriodicInZ::type is_periodic_in_z,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map) noexcept
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),          // NOLINT
      outer_radius_(std::move(outer_radius)),          // NOLINT
      lower_bound_(std::move(lower_bound)),            // NOLINT
      upper_bound_(std::move(upper_bound)),            // NOLINT
      is_periodic_in_z_(std::move(is_periodic_in_z)),  // NOLINT
      initial_refinement_(                             // NOLINT
          std::move(initial_refinement)),              // NOLINT
      initial_number_of_grid_points_(                  // NOLINT
          std::move(initial_number_of_grid_points)),   // NOLINT
      use_equiangular_map_(use_equiangular_map) {}     // NOLINT

template <typename TargetFrame>
Domain<3, TargetFrame> Cylinder<TargetFrame>::create_domain() const noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3DPrism =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
  using Wedge2D = CoordinateMaps::Wedge2D;
  using Wedge3DPrism = CoordinateMaps::ProductOf2Maps<Wedge2D, Affine>;

  std::vector<std::array<size_t, 8>> corners{
      {{1, 5, 3, 7, 9, 13, 11, 15}},   //+x wedge
      {{3, 7, 2, 6, 11, 15, 10, 14}},  //+y wedge
      {{2, 6, 0, 4, 10, 14, 8, 12}},   //-x wedge
      {{0, 4, 1, 5, 8, 12, 9, 13}},    //-y wedge
      {{0, 1, 2, 3, 8, 9, 10, 11}}};   // Center square prism

  auto coord_maps =
      make_vector_coordinate_map_base<Frame::Logical, TargetFrame>(
          Wedge3DPrism{Wedge2D{inner_radius_, outer_radius_, 0.0, 1.0,
                               OrientationMap<2>{std::array<Direction<2>, 2>{
                                   {Direction<2>::upper_xi(),
                                    Direction<2>::upper_eta()}}},
                               use_equiangular_map_},
                       Affine{-1.0, 1.0, lower_bound_, upper_bound_}},
          Wedge3DPrism{Wedge2D{inner_radius_, outer_radius_, 0.0, 1.0,
                               OrientationMap<2>{std::array<Direction<2>, 2>{
                                   {Direction<2>::lower_eta(),
                                    Direction<2>::upper_xi()}}},
                               use_equiangular_map_},
                       Affine{-1.0, 1.0, lower_bound_, upper_bound_}},
          Wedge3DPrism{Wedge2D{inner_radius_, outer_radius_, 0.0, 1.0,
                               OrientationMap<2>{std::array<Direction<2>, 2>{
                                   {Direction<2>::lower_xi(),
                                    Direction<2>::lower_eta()}}},
                               use_equiangular_map_},
                       Affine{-1.0, 1.0, lower_bound_, upper_bound_}},
          Wedge3DPrism{Wedge2D{inner_radius_, outer_radius_, 0.0, 1.0,
                               OrientationMap<2>{std::array<Direction<2>, 2>{
                                   {Direction<2>::upper_eta(),
                                    Direction<2>::lower_xi()}}},
                               use_equiangular_map_},
                       Affine{-1.0, 1.0, lower_bound_, upper_bound_}});

  if (use_equiangular_map_) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Equiangular3DPrism{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0)),
                Affine{-1.0, 1.0, lower_bound_, upper_bound_}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(2.0),
                            inner_radius_ / sqrt(2.0)),
                     Affine{-1.0, 1.0, lower_bound_, upper_bound_}}));
  }
  return Domain<3, TargetFrame>{
      std::move(coord_maps), corners,
      is_periodic_in_z_
          ? std::vector<PairOfFaces>{{{0, 1, 2, 3}, {8, 9, 10, 11}},
                                     {{1, 5, 3, 7}, {9, 13, 11, 15}},
                                     {{4, 5, 0, 1}, {12, 13, 8, 9}},
                                     {{4, 0, 6, 2}, {12, 8, 14, 10}},
                                     {{2, 3, 6, 7}, {10, 11, 14, 15}}}
          : std::vector<PairOfFaces>{}};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 3>> Cylinder<TargetFrame>::initial_extents()
    const noexcept {
  return {
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      initial_number_of_grid_points_,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[2]}}};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 3>>
Cylinder<TargetFrame>::initial_refinement_levels() const noexcept {
  return {5, make_array<3>(initial_refinement_)};
}

template class Cylinder<Frame::Grid>;
template class Cylinder<Frame::Inertial>;
}  // namespace creators
}  // namespace domain
