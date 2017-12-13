// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/Shell.hpp"

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeArray.hpp"

namespace DomainCreators {

template <typename TargetFrame>
Shell<TargetFrame>::Shell(
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
Domain<3, TargetFrame> Shell<TargetFrame>::create_domain() const noexcept {
  using Wedge3DMap = CoordinateMaps::Wedge3D;

  std::vector<std::array<size_t, 8>> corners{
      {{7, 5, 8, 6, 15, 13, 16, 14}},   // Upper z
      {{4, 2, 3, 1, 12, 10, 11, 9}},    // Lower z
      {{4, 8, 2, 6, 12, 16, 10, 14}},   // Upper y
      {{7, 3, 5, 1, 15, 11, 13, 9}},    // Lower y
      {{1, 2, 5, 6, 9, 10, 13, 14}},    // Upper x
      {{4, 3, 8, 7, 12, 11, 16, 15}}};  // Lower x

  return Domain<3, TargetFrame>{
      make_vector_coordinate_map_base<Frame::Logical, TargetFrame>(
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::upper_zeta(),
                     1.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::lower_zeta(),
                     1.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::upper_eta(),
                     1.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::lower_eta(),
                     1.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::upper_xi(),
                     1.0, use_equiangular_map_},
          Wedge3DMap{inner_radius_, outer_radius_, Direction<3>::lower_xi(),
                     1.0, use_equiangular_map_}),
      corners};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 3>> Shell<TargetFrame>::initial_extents() const
    noexcept {
  return {
      6,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
}
template <typename TargetFrame>
std::vector<std::array<size_t, 3>>
Shell<TargetFrame>::initial_refinement_levels() const noexcept {
  return {6, make_array<3>(initial_refinement_)};
}
}  // namespace DomainCreators

template class DomainCreators::Shell<Frame::Grid>;
template class DomainCreators::Shell<Frame::Inertial>;
