// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace creators {

/// \ingroup DomainCreatorsGroup
/// \brief Create a Domain consisting of multiple aligned Blocks arrayed in a
/// lattice.
///
/// This is useful for setting up problems with piecewise smooth initial data,
/// problems that specify different boundary conditions on distinct parts of
/// the boundary, or problems that need different length scales initially.
///
/// \note Adaptive mesh refinement can never join Block%s, so use the fewest
/// number of Block%s that your problem needs.  More initial Element%s can be
/// created by specifying a larger `InitialRefinement`.
template <size_t VolumeDim, typename TargetFrame>
class AlignedLattice : public DomainCreator<VolumeDim, TargetFrame> {
 public:
  struct BlockBounds {
    using type = std::array<std::vector<double>, VolumeDim>;
    static constexpr OptionString help = {
        "Coordinates of block boundaries in each dimension."};
  };

  struct IsPeriodicIn {
    using type = std::array<bool, VolumeDim>;
    static constexpr OptionString help = {
        "Whether the domain is periodic in each dimension."};
    static type default_value() noexcept {
      return make_array<VolumeDim>(false);
    }
  };

  struct InitialRefinement {
    using type = std::array<size_t, VolumeDim>;
    static constexpr OptionString help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, VolumeDim>;
    static constexpr OptionString help = {
        "Initial number of grid points in each dimension."};
  };

  struct BlocksToExclude {
    using type = std::vector<std::array<size_t, VolumeDim>>;
    static constexpr OptionString help = {
        "List of Block indices to exclude, if any."};
    static type default_value() noexcept {
      return std::vector<std::array<size_t, VolumeDim>>{};
    }
  };

  using options = tmpl::list<BlockBounds, IsPeriodicIn, InitialRefinement,
                             InitialGridPoints, BlocksToExclude>;

  static constexpr OptionString help = {
      "AlignedLattice creates a regular lattice of blocks whose corners are\n"
      "given by tensor products of the specified BlockBounds. Each Block in\n"
      "the lattice is identified by a VolumeDim-tuple of zero-based indices\n"
      "Supplying a list of these tuples to BlocksToExclude will result in\n"
      "the domain having the corresponding Blocks excluded. See the Domain\n"
      "Creation tutorial in the documentation for more information on Block\n"
      "numberings in rectilinear domains. Note that if any Blocks are\n"
      "excluded, setting the option IsPeriodicIn to `true` in any dimension\n"
      "will trigger an error, as periodic boundary\n"
      "conditions for this domain with holes is not supported."};

  AlignedLattice(typename BlockBounds::type block_bounds,
                 typename IsPeriodicIn::type is_periodic_in,
                 typename InitialRefinement::type initial_refinement_levels,
                 typename InitialGridPoints::type initial_number_of_grid_points,
                 typename BlocksToExclude::type blocks_to_exclude) noexcept;

  AlignedLattice() = default;
  AlignedLattice(const AlignedLattice&) = delete;
  AlignedLattice(AlignedLattice&&) noexcept = default;
  AlignedLattice& operator=(const AlignedLattice&) = delete;
  AlignedLattice& operator=(AlignedLattice&&) noexcept = default;
  ~AlignedLattice() override = default;

  Domain<VolumeDim, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, VolumeDim>> initial_extents() const
      noexcept override;

  std::vector<std::array<size_t, VolumeDim>> initial_refinement_levels() const
      noexcept override;

 private:
  typename BlockBounds::type block_bounds_{
      make_array<VolumeDim, std::vector<double>>({})};
  typename IsPeriodicIn::type is_periodic_in_{make_array<VolumeDim>(false)};
  typename InitialRefinement::type initial_refinement_levels_{
      make_array<VolumeDim>(std::numeric_limits<size_t>::max())};
  typename InitialGridPoints::type initial_number_of_grid_points_{
      make_array<VolumeDim>(std::numeric_limits<size_t>::max())};
  typename BlocksToExclude::type blocks_to_exclude_{};
  Index<VolumeDim> number_of_blocks_by_dim_{};
};
}  // namespace creators
}  // namespace domain
