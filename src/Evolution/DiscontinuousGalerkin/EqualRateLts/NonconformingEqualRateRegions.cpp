// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/NonconformingEqualRateRegions.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.tpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg {
template <size_t Dim>
NonconformingEqualRateRegions<Dim>::NonconformingEqualRateRegions(
    const std::unique_ptr<DomainCreator<Dim>>& domain_creator) {
  const Domain<Dim> domain = domain_creator->create_domain();
  const auto initial_levels = domain_creator->initial_refinement_levels();

  // Find all the non-conforming boundaries
  for (const auto& block : domain.blocks()) {
    std::vector<std::vector<std::pair<size_t, Direction<Dim>>>> boundaries{};
    for (const auto& [direction, neighbors] : block.neighbors()) {
      if (neighbors.are_conforming() or neighbors.size() == 1) {
        continue;
      }
      std::vector<std::pair<size_t, Direction<Dim>>> boundaries_for_direction{};
      boundaries_for_direction.reserve(neighbors.size() + 1);
      boundaries_for_direction.emplace_back(block.id(), direction);
      for (const auto& [neighbor, orientation] : neighbors.orientations()) {
        boundaries_for_direction.emplace_back(
            neighbor, orientation(direction).opposite());
      }
      boundaries.push_back(std::move(boundaries_for_direction));
    }

    if (boundaries.empty()) {
      continue;
    }

    if (boundaries.size() == 2 and
        // First element is the current block
        boundaries[0].front().second.dimension() ==
            boundaries[1].front().second.dimension() and
        initial_levels[block.id()][boundaries[0].front().second.dimension()] >
            0) {
      // Nonconforming regions on two sides of the block that are not
      // adjacent to any of the same elements.
      for (auto& boundary : boundaries) {
        region_names_.push_back(
            "Nonconforming" + std::to_string(block.id()) +
            (boundary.front().second.side() == Side::Upper ? "+" : "-"));
        regions_.push_back(std::move(boundary));
      }
    } else {
      // Either there is only one nonconforming side or the
      // nonconforming sides share elements, so they all form one
      // region.
      size_t full_size = 0;
      for (const auto& boundary : boundaries) {
        full_size += boundary.size();
      }
      auto boundary = std::move(boundaries.front());
      boundary.reserve(full_size);
      for (size_t i = 1; i < boundaries.size(); ++i) {
        boundary.insert(boundary.end(), boundaries[i].begin(),
                        boundaries[i].end());
      }
      region_names_.push_back("Nonconforming" + std::to_string(block.id()));
      regions_.push_back(std::move(boundary));
    }
  }
}

template <size_t Dim>
std::unordered_map<std::string, size_t>
NonconformingEqualRateRegions<Dim>::regions() const {
  std::unordered_map<std::string, size_t> result{};
  for (size_t i = 0; i < region_names_.size(); ++i) {
    const bool inserted = result.emplace(region_names_[i], i).second;
    if (not inserted) {
      ERROR("Generated two regions named " << region_names_[i]);
    }
  }
  return result;
}

template <size_t Dim>
bool NonconformingEqualRateRegions<Dim>::is_in_region(
    const size_t region, const ElementId<Dim>& element_id) const {
  return alg::any_of(regions_[region], [&](const auto& region_info) {
    const auto& [block_id, direction] = region_info;
    if (element_id.block_id() != block_id) {
      return false;
    }
    const auto perpendicular_segment =
        element_id.segment_id(direction.dimension());
    return perpendicular_segment.index() ==
           (direction.side() == Side::Lower
                ? 0
                : two_to_the(perpendicular_segment.refinement_level()) - 1);
  });
}

template <size_t Dim>
void NonconformingEqualRateRegions<Dim>::pup(PUP::er& p) {
  p | region_names_;
  p | regions_;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template class NonconformingEqualRateRegions<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)       \
  template class EqualRateRegions< \
      DIM(data), tmpl::list<NonconformingEqualRateRegions<DIM(data)>>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg
