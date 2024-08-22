// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <limits>
#include <memory>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {

template <size_t Dim>
struct RefinementRegion {
  std::array<size_t, Dim> lower_corner_index;
  std::array<size_t, Dim> upper_corner_index;
  std::array<size_t, Dim> refinement;

  struct LowerCornerIndex {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {"Lower bound of refined region."};
  };

  struct UpperCornerIndex {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {"Upper bound of refined region."};
  };

  struct Refinement {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {"Refinement inside region."};
  };

  static constexpr Options::String help = {
      "A region to be refined differently from the default for the lattice.\n"
      "The region is a box between the block boundaries indexed by the\n"
      "Lower- and UpperCornerIndex options."};
  using options = tmpl::list<LowerCornerIndex, UpperCornerIndex, Refinement>;
  RefinementRegion(const std::array<size_t, Dim>& lower_corner_index_in,
                   const std::array<size_t, Dim>& upper_corner_index_in,
                   const std::array<size_t, Dim>& refinement_in)
      : lower_corner_index(lower_corner_index_in),
        upper_corner_index(upper_corner_index_in),
        refinement(refinement_in) {}
  RefinementRegion() = default;
};

/// \cond
// This is needed to print the default value for the RefinedGridPoints
// option.  Since the default value is an empty vector, this function
// is never actually called.
template <size_t Dim>
[[noreturn]] std::ostream& operator<<(std::ostream& /*s*/,
                                      const RefinementRegion<Dim>& /*unused*/);
/// \endcond

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
template <size_t Dim>
class AlignedLattice : public DomainCreator<Dim> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Affine>,
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>>;

  struct BlockBounds {
    using type = std::array<std::vector<double>, Dim>;
    static constexpr Options::String help = {
        "Coordinates of block boundaries in each dimension."};
  };

  struct IsPeriodicIn {
    using type = std::array<bool, Dim>;
    static constexpr Options::String help = {
        "Whether the domain is periodic in each dimension."};
  };

  struct InitialLevels {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {
        "Initial number of grid points in each dimension."};
  };

  struct RefinedLevels {
    using type = std::vector<RefinementRegion<Dim>>;
    static constexpr Options::String help = {
        "h-refined regions.  Later entries take priority."};
  };

  struct RefinedGridPoints {
    using type = std::vector<RefinementRegion<Dim>>;
    static constexpr Options::String help = {
        "p-refined regions.  Later entries take priority."};
  };

  struct BlocksToExclude {
    using type = std::vector<std::array<size_t, Dim>>;
    static constexpr Options::String help = {
        "List of Block indices to exclude, if any."};
  };

  template <typename Metavariables>
  using options = tmpl::list<
      BlockBounds, InitialLevels, InitialGridPoints, RefinedLevels,
      RefinedGridPoints, BlocksToExclude,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          typename Rectilinear<Dim>::template BoundaryConditions<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>,
          IsPeriodicIn>>;

  static constexpr Options::String help = {
      "AlignedLattice creates a regular lattice of blocks whose corners are\n"
      "given by tensor products of the specified BlockBounds. Each Block in\n"
      "the lattice is identified by a Dim-tuple of zero-based indices\n"
      "Supplying a list of these tuples to BlocksToExclude will result in\n"
      "the domain having the corresponding Blocks excluded. See the Domain\n"
      "Creation tutorial in the documentation for more information on Block\n"
      "numberings in rectilinear domains. Note that if any Blocks are\n"
      "excluded, setting the option IsPeriodicIn to `true` in any dimension\n"
      "will trigger an error, as periodic boundary\n"
      "conditions for this domain with holes is not supported."};

  AlignedLattice(std::array<std::vector<double>, Dim> block_bounds,
                 std::array<size_t, Dim> initial_refinement_levels,
                 std::array<size_t, Dim> initial_number_of_grid_points,
                 std::vector<RefinementRegion<Dim>> refined_refinement,
                 std::vector<RefinementRegion<Dim>> refined_grid_points,
                 std::vector<std::array<size_t, Dim>> blocks_to_exclude,
                 std::array<bool, Dim> is_periodic_in = make_array<Dim>(false),
                 const Options::Context& context = {});

  AlignedLattice(
      std::array<std::vector<double>, Dim> block_bounds,
      std::array<size_t, Dim> initial_refinement_levels,
      std::array<size_t, Dim> initial_number_of_grid_points,
      std::vector<RefinementRegion<Dim>> refined_refinement,
      std::vector<RefinementRegion<Dim>> refined_grid_points,
      std::vector<std::array<size_t, Dim>> blocks_to_exclude,
      std::array<std::array<std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>,
                            2>,
                 Dim>
          boundary_conditions,
      const Options::Context& context = {});

  template <typename BoundaryConditionsBase>
  AlignedLattice(
      std::array<std::vector<double>, Dim> block_bounds,
      std::array<size_t, Dim> initial_refinement_levels,
      std::array<size_t, Dim> initial_number_of_grid_points,
      std::vector<RefinementRegion<Dim>> refined_refinement,
      std::vector<RefinementRegion<Dim>> refined_grid_points,
      std::vector<std::array<size_t, Dim>> blocks_to_exclude,
      std::array<std::variant<std::unique_ptr<BoundaryConditionsBase>,
                              typename Rectilinear<Dim>::
                                  template LowerUpperBoundaryCondition<
                                      BoundaryConditionsBase>>,
                 Dim>
          boundary_conditions,
      const Options::Context& context = {})
      : AlignedLattice(std::move(block_bounds), initial_refinement_levels,
                       initial_number_of_grid_points, refined_refinement,
                       refined_grid_points, blocks_to_exclude,
                       Rectilinear<Dim>::transform_boundary_conditions(
                           std::move(boundary_conditions)),
                       context) {}

  AlignedLattice() = default;
  AlignedLattice(const AlignedLattice&) = delete;
  AlignedLattice(AlignedLattice&&) = default;
  AlignedLattice& operator=(const AlignedLattice&) = delete;
  AlignedLattice& operator=(AlignedLattice&&) = default;
  ~AlignedLattice() override = default;

  Domain<Dim> create_domain() const override;

  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, Dim>> initial_extents() const override;

  std::vector<std::array<size_t, Dim>> initial_refinement_levels()
      const override;

 private:
  std::array<std::vector<double>, Dim> block_bounds_{
      make_array<Dim, std::vector<double>>({})};
  std::array<bool, Dim> is_periodic_in_{make_array<Dim>(false)};
  std::array<size_t, Dim> initial_refinement_levels_{
      make_array<Dim>(std::numeric_limits<size_t>::max())};
  std::array<size_t, Dim> initial_number_of_grid_points_{
      make_array<Dim>(std::numeric_limits<size_t>::max())};
  std::vector<RefinementRegion<Dim>> refined_refinement_{};
  std::vector<RefinementRegion<Dim>> refined_grid_points_{};
  std::vector<std::array<size_t, Dim>> blocks_to_exclude_{};
  Index<Dim> number_of_blocks_by_dim_{};
  std::array<
      std::array<std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
                 2>,
      Dim>
      boundary_conditions_{};
};
}  // namespace creators
}  // namespace domain
