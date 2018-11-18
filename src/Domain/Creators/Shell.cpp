// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Shell.hpp"

#include <memory>
#include <utility>

#include "Domain/Block.hpp"                         // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"                 // IWYU pragma: keep
#include "Domain/CoordinateMaps/CoordinateMap.hpp"  // IWYU pragma: keep
#include "Domain/Creators/DomainCreator.hpp"        // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Utilities/Gsl.hpp"
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
Shell<TargetFrame>::Shell(
    typename InnerRadius::type inner_radius,
    typename OuterRadius::type outer_radius,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map,
    typename AspectRatio::type aspect_ratio,
    typename UseLogarithmicMap::type use_logarithmic_map,
    typename WhichWedges::type which_wedges,
    typename RadialBlockLayers::type number_of_layers) noexcept
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),                // NOLINT
      outer_radius_(std::move(outer_radius)),                // NOLINT
      initial_refinement_(                                   // NOLINT
          std::move(initial_refinement)),                    // NOLINT
      initial_number_of_grid_points_(                        // NOLINT
          std::move(initial_number_of_grid_points)),         // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)),  // NOLINT
      aspect_ratio_(std::move(aspect_ratio)),                // NOLINT
      use_logarithmic_map_(std::move(use_logarithmic_map)),  // NOLINT
      which_wedges_(std::move(which_wedges)),                // NOLINT
      number_of_layers_(std::move(number_of_layers)) {}      // NOLINT

template <typename TargetFrame>
Domain<3, TargetFrame> Shell<TargetFrame>::create_domain() const noexcept {
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
      coord_maps = wedge_coordinate_maps<TargetFrame>(
          inner_radius_, outer_radius_, 1.0, 1.0, use_equiangular_map_, 0.0,
          false, aspect_ratio_, use_logarithmic_map_, which_wedges_,
          number_of_layers_);
  return Domain<3, TargetFrame>{
      std::move(coord_maps),
      corners_for_radially_layered_domains(
          number_of_layers_, false, {{1, 2, 3, 4, 5, 6, 7, 8}}, which_wedges_)};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 3>> Shell<TargetFrame>::initial_extents() const
    noexcept {
  std::vector<std::array<size_t, 3>>::size_type num_wedges =
      6 * number_of_layers_;
  if (UNLIKELY(which_wedges_ == ShellWedges::FourOnEquator)) {
    num_wedges = 4 * number_of_layers_;
  } else if (UNLIKELY(which_wedges_ == ShellWedges::OneAlongMinusX)) {
    num_wedges = number_of_layers_;
  }
  return {
      num_wedges,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
}
template <typename TargetFrame>
std::vector<std::array<size_t, 3>>
Shell<TargetFrame>::initial_refinement_levels() const noexcept {
  std::vector<std::array<size_t, 3>>::size_type num_wedges =
      6 * number_of_layers_;
  if (UNLIKELY(which_wedges_ == ShellWedges::FourOnEquator)) {
    num_wedges = 4 * number_of_layers_;
  } else if (UNLIKELY(which_wedges_ == ShellWedges::OneAlongMinusX)) {
    num_wedges = number_of_layers_;
  }
  return {num_wedges, make_array<3>(initial_refinement_)};
}

template class Shell<Frame::Grid>;
template class Shell<Frame::Inertial>;
}  // namespace creators
}  // namespace domain
