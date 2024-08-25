// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/AlignedLattice.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::creators {

using ::operator<<;
template <size_t Dim>
std::ostream& operator<<(std::ostream& /*s*/,
                         const RefinementRegion<Dim>& /*unused*/) {
  ERROR(
      "RefinementRegion stream operator is only for option parsing and "
      "should never be called.");
}

template <size_t Dim>
AlignedLattice<Dim>::AlignedLattice(
    std::array<std::vector<double>, Dim> block_bounds,
    std::array<size_t, Dim> initial_refinement_levels,
    std::array<size_t, Dim> initial_number_of_grid_points,
    std::vector<RefinementRegion<Dim>> refined_refinement,
    std::vector<RefinementRegion<Dim>> refined_grid_points,
    std::vector<std::array<size_t, Dim>> blocks_to_exclude,
    std::array<bool, Dim> is_periodic_in, const Options::Context& context)
    : block_bounds_(std::move(block_bounds)),
      is_periodic_in_(is_periodic_in),
      initial_refinement_levels_(initial_refinement_levels),
      initial_number_of_grid_points_(initial_number_of_grid_points),
      refined_refinement_(std::move(refined_refinement)),
      refined_grid_points_(std::move(refined_grid_points)),
      blocks_to_exclude_(std::move(blocks_to_exclude)),
      number_of_blocks_by_dim_{map_array(
          block_bounds_,
          [](const std::vector<double>& v) { return v.size() - 1; })} {
  if (not blocks_to_exclude_.empty() and
      alg::any_of(is_periodic_in_, [](const bool t) { return t; })) {
    PARSE_ERROR(context,
                "Cannot exclude blocks as well as have periodic boundary "
                "conditions!");
  }
  for (const auto& refinement_region : refined_grid_points_) {
    for (size_t i = 0; i < Dim; ++i) {
      if (gsl::at(refinement_region.upper_corner_index, i) >=
          gsl::at(block_bounds_, i).size()) {
        PARSE_ERROR(context, "Refinement region extends to "
                                 << refinement_region.upper_corner_index
                                 << ", which is outside the domain");
      }
    }
  }
  for (const auto& refinement_region : refined_refinement_) {
    for (size_t i = 0; i < Dim; ++i) {
      if (gsl::at(refinement_region.upper_corner_index, i) >=
          gsl::at(block_bounds_, i).size()) {
        PARSE_ERROR(context, "Refinement region extends to "
                                 << refinement_region.upper_corner_index
                                 << ", which is outside the domain");
      }
    }
  }
}

template <size_t Dim>
AlignedLattice<Dim>::AlignedLattice(
    std::array<std::vector<double>, Dim> block_bounds,
    std::array<size_t, Dim> initial_refinement_levels,
    std::array<size_t, Dim> initial_number_of_grid_points,
    std::vector<RefinementRegion<Dim>> refined_refinement,
    std::vector<RefinementRegion<Dim>> refined_grid_points,
    std::vector<std::array<size_t, Dim>> blocks_to_exclude,
    std::array<
        std::array<
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>, 2>,
        Dim>
        boundary_conditions,
    const Options::Context& context)
    : AlignedLattice(
          std::move(block_bounds), initial_refinement_levels,
          initial_number_of_grid_points, std::move(refined_refinement),
          std::move(refined_grid_points), std::move(blocks_to_exclude),
          make_array<Dim>(false), context) {
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  boundary_conditions_ = std::move(boundary_conditions);
  using domain::BoundaryConditions::is_none;
  using domain::BoundaryConditions::is_periodic;
  for (size_t d = 0; d < Dim; ++d) {
    const auto& [lower_bc, upper_bc] = gsl::at(boundary_conditions_, d);
    ASSERT(lower_bc != nullptr and upper_bc != nullptr,
           "None of the boundary conditions can be nullptr.");
    if (is_none(lower_bc) or is_none(upper_bc)) {
      PARSE_ERROR(
          context,
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that.");
    }
    if (is_periodic(lower_bc) != is_periodic(upper_bc)) {
      PARSE_ERROR(context,
                  "Periodic boundary conditions must be applied for both "
                  "upper and lower direction in a dimension.");
    }
    if (is_periodic(lower_bc) and is_periodic(upper_bc)) {
      gsl::at(is_periodic_in_, d) = true;
    }
  }
  if (not blocks_to_exclude_.empty() and
      alg::any_of(is_periodic_in_, [](const bool t) { return t; })) {
    PARSE_ERROR(context,
                "Cannot exclude blocks as well as have periodic boundary "
                "conditions!");
  }
  for (const auto& refinement_region : refined_grid_points_) {
    for (size_t i = 0; i < Dim; ++i) {
      if (gsl::at(refinement_region.upper_corner_index, i) >=
          gsl::at(block_bounds_, i).size()) {
        PARSE_ERROR(context, "Refinement region extends to "
                                 << refinement_region.upper_corner_index
                                 << ", which is outside the domain");
      }
    }
  }
  for (const auto& refinement_region : refined_refinement_) {
    for (size_t i = 0; i < Dim; ++i) {
      if (gsl::at(refinement_region.upper_corner_index, i) >=
          gsl::at(block_bounds_, i).size()) {
        PARSE_ERROR(context, "Refinement region extends to "
                                 << refinement_region.upper_corner_index
                                 << ", which is outside the domain");
      }
    }
  }
}

template <size_t Dim>
Domain<Dim> AlignedLattice<Dim>::create_domain() const {
  return rectilinear_domain<Dim>(
      number_of_blocks_by_dim_, block_bounds_,
      {std::vector<Index<Dim>>(blocks_to_exclude_.begin(),
                               blocks_to_exclude_.end())},
      {}, is_periodic_in_, {}, false);
}

template <size_t Dim>
std::vector<DirectionMap<
    Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
AlignedLattice<Dim>::external_boundary_conditions() const {
  if (boundary_conditions_[0][0] == nullptr) {
#ifdef SPECTRE_DEBUG
    for (size_t d = 0; d < Dim; ++d) {
      ASSERT(gsl::at(boundary_conditions_, d)[0] == nullptr and
                 gsl::at(boundary_conditions_, d)[1] == nullptr,
             "Boundary conditions must be set for all directions or none.");
    }
#endif  // SPECTRE_DEBUG
    return {};
  }
  // Set boundary conditions by using the computed domain's external
  // boundaries
  const auto domain = create_domain();
  const auto& blocks = domain.blocks();
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{blocks.size()};
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (const Direction<Dim>& external_direction :
         blocks[i].external_boundaries()) {
      boundary_conditions[i][external_direction] =
          gsl::at(gsl::at(boundary_conditions_, external_direction.dimension()),
                  external_direction.side() == Side::Lower ? 0 : 1)
              ->get_clone();
    }
  }
  return boundary_conditions;
}

namespace {
template <size_t Dim>
std::vector<std::array<size_t, Dim>> apply_refinement_regions(
    const Index<Dim>& number_of_blocks_by_dim,
    const std::vector<std::array<size_t, Dim>>& blocks_to_exclude,
    const std::array<size_t, Dim>& default_refinement,
    const std::vector<RefinementRegion<Dim>>& refinement_regions) {
  std::vector<std::array<size_t, Dim>> result;
  for (const auto& block_index : indices_for_rectilinear_domains(
           number_of_blocks_by_dim,
           std::vector<Index<Dim>>(blocks_to_exclude.begin(),
                                   blocks_to_exclude.end()))) {
    std::array<size_t, Dim> block_result = default_refinement;
    for (const auto& refinement_region : refinement_regions) {
      for (size_t d = 0; d < Dim; ++d) {
        if (block_index[d] < gsl::at(refinement_region.lower_corner_index, d) or
            block_index[d] >=
                gsl::at(refinement_region.upper_corner_index, d)) {
          goto next_region;
        }
      }
      block_result = refinement_region.refinement;
    next_region:;
    }
    result.push_back(block_result);
  }
  return result;
}
}  // namespace

template <size_t Dim>
std::vector<std::array<size_t, Dim>> AlignedLattice<Dim>::initial_extents()
    const {
  return apply_refinement_regions(number_of_blocks_by_dim_, blocks_to_exclude_,
                                  initial_number_of_grid_points_,
                                  refined_grid_points_);
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>>
AlignedLattice<Dim>::initial_refinement_levels() const {
  return apply_refinement_regions(number_of_blocks_by_dim_, blocks_to_exclude_,
                                  initial_refinement_levels_,
                                  refined_refinement_);
}

template class AlignedLattice<1>;
template class AlignedLattice<2>;
template class AlignedLattice<3>;
template std::ostream& operator<<(std::ostream& /*s*/,
                                  const RefinementRegion<1>& /*unused*/);
template std::ostream& operator<<(std::ostream& /*s*/,
                                  const RefinementRegion<2>& /*unused*/);
template std::ostream& operator<<(std::ostream& /*s*/,
                                  const RefinementRegion<3>& /*unused*/);
}  // namespace domain::creators
