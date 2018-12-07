// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Sphere.hpp"

#include <cmath>
#include <memory>

#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace Frame {
struct Grid;      // IWYU pragma: keep
struct Inertial;  // IWYU pragma: keep
struct Logical;   // IWYU pragma: keep
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {

template <typename TargetFrame>
Sphere<TargetFrame>::Sphere(
    typename InnerRadius::type inner_radius,
    typename OuterRadius::type outer_radius,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map) noexcept
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),                  // NOLINT
      outer_radius_(std::move(outer_radius)),                  // NOLINT
      initial_refinement_(                                     // NOLINT
          std::move(initial_refinement)),                      // NOLINT
      initial_number_of_grid_points_(                          // NOLINT
          std::move(initial_number_of_grid_points)),           // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)) {}  // NOLINT

template <typename TargetFrame>
Domain<3, TargetFrame> Sphere<TargetFrame>::create_domain() const noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
  std::vector<std::array<size_t, 8>> corners =
      corners_for_radially_layered_domains(1, true);

  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
      coord_maps = wedge_coordinate_maps<TargetFrame>(
          inner_radius_, outer_radius_, 0.0, 1.0, use_equiangular_map_);
  if (use_equiangular_map_) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Equiangular3D{
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                        inner_radius_ / sqrt(3.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                        inner_radius_ / sqrt(3.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                        inner_radius_ / sqrt(3.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0))}));
  }
  return Domain<3, TargetFrame>(std::move(coord_maps), corners);
}

template <typename TargetFrame>
std::vector<std::array<size_t, 3>> Sphere<TargetFrame>::initial_extents() const
    noexcept {
  std::vector<std::array<size_t, 3>> extents{
      6,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
  extents.push_back(
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[1]}});
  return extents;
}
template <typename TargetFrame>
std::vector<std::array<size_t, 3>>
Sphere<TargetFrame>::initial_refinement_levels() const noexcept {
  return {7, make_array<3>(initial_refinement_)};
}

template class Sphere<Frame::Grid>;
template class Sphere<Frame::Inertial>;
}  // namespace creators
}  // namespace domain
