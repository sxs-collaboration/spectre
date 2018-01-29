// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/Sphere.hpp"

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/MakeArray.hpp"

namespace DomainCreators {

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
  using Wedge3DMap = CoordinateMaps::Wedge3D;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

  std::vector<std::array<size_t, 8>> corners{
      {{7, 5, 8, 6, 15, 13, 16, 14}},  // Upper z
      {{4, 2, 3, 1, 12, 10, 11, 9}},   // Lower z
      {{4, 8, 2, 6, 12, 16, 10, 14}},  // Upper y
      {{7, 3, 5, 1, 15, 11, 13, 9}},   // Lower y
      {{1, 2, 5, 6, 9, 10, 13, 14}},   // Upper x
      {{4, 3, 8, 7, 12, 11, 16, 15}},  // Lower x
      {{3, 1, 4, 2, 7, 5, 8, 6}}};     // center cube

  auto coord_maps =
      make_vector_coordinate_map_base<Frame::Logical, TargetFrame>(
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::upper_zeta(),
                     0.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::lower_zeta(),
                     0.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::upper_eta(),
                     0.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::lower_eta(),
                     0.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::upper_xi(),
                     0.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::lower_xi(),
                     0.0, use_equiangular_map_});
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
}  // namespace DomainCreators

template class DomainCreators::Sphere<Frame::Grid>;
template class DomainCreators::Sphere<Frame::Inertial>;
