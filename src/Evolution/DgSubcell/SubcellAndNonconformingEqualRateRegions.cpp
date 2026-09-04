// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/SubcellAndNonconformingEqualRateRegions.hpp"

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
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.tpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
template <size_t Dim>
SubcellAndNonconformingEqualRateRegions<Dim>::
    SubcellAndNonconformingEqualRateRegions(
        const SubcellOptions& subcell_options,
        const std::unique_ptr<DomainCreator<Dim>>& domain_creator) {
  // Reconstruct the  SubcellOptions to get the auto-detected
  // only_dg_block_ids (non-hypercube topology blocks plus user-specified
  // OnlyDgBlocksAndGroups).
  const auto real_subcell_options =
      Tags::SubcellOptions<Dim>::create_from_options(subcell_options,
                                                     domain_creator);
  only_dg_block_ids_ = real_subcell_options.only_dg_block_ids();

  const Domain<Dim> domain = domain_creator->create_domain();
  const auto initial_levels = domain_creator->initial_refinement_levels();

  // Categorize each nonconforming interface (a block face with more than one
  // neighbor) as either:
  //   - subcell-adjacent: at least one block on either side is subcell-capable.
  //     The DG-only blocks at this face are added to subcell_adjacent_dg_faces_
  //     so their outermost elements join the Subcell region.
  //   - purely DG: all blocks on both sides are DG-only.
  //     Becomes a separate NonconformingN[+/-] region, identical to
  //     NonconformingEqualRateRegions.
  //
  // Note: a DG-only block may appear at two nonconforming interfaces, one
  // subcell-adjacent and one purely DG, placing it in both
  // subcell_adjacent_dg_faces_ and a NonconformingN region.  Whether this
  // creates an element-level conflict depends on the refinement and axis of
  // the two faces.  A post-processing pass below resolves conflicts by merging
  // the NonconformingN region into Subcell; when no conflict exists the region
  // is kept as an independent LTS region.
  //
  // The loop structure mirrors NonconformingEqualRateRegions: it fires only for
  // the "one" side (the block with multiple neighbors on a single face), since
  // the "many" side blocks each see only one conforming neighbor.
  for (const auto& block : domain.blocks()) {
    // Collect all nonconforming boundaries of this block.
    std::vector<std::vector<std::pair<size_t, Direction<Dim>>>> boundaries{};

    for (const auto& [direction, neighbors] : block.neighbors()) {
      if (neighbors.are_conforming() or neighbors.size() == 1) {
        continue;
      }

      // Determine whether any block at this interface supports subcell.
      // Skip the neighbor scan if the current block is already subcell-capable.
      bool any_subcell_capable = not alg::found(only_dg_block_ids_, block.id());
      if (not any_subcell_capable) {
        for (const auto& [neighbor_id, orientation] :
             neighbors.orientations()) {
          if (not alg::found(only_dg_block_ids_, neighbor_id)) {
            any_subcell_capable = true;
            break;
          }
        }
      }

      if (any_subcell_capable) {
        // Subcell-adjacent interface: record the DG-only faces to fold into
        // the Subcell region.
        if (alg::found(only_dg_block_ids_, block.id())) {
          subcell_adjacent_dg_faces_.emplace_back(block.id(), direction);
        }
        for (const auto& [neighbor_id, orientation] :
             neighbors.orientations()) {
          if (alg::found(only_dg_block_ids_, neighbor_id)) {
            subcell_adjacent_dg_faces_.emplace_back(
                neighbor_id, orientation(direction).opposite());
          }
        }
      } else {
        // Purely DG nonconforming interface: collect for region creation.
        std::vector<std::pair<size_t, Direction<Dim>>>
            boundaries_for_direction{};
        boundaries_for_direction.reserve(neighbors.size() + 1);
        boundaries_for_direction.emplace_back(block.id(), direction);
        for (const auto& [neighbor_id, orientation] :
             neighbors.orientations()) {
          boundaries_for_direction.emplace_back(
              neighbor_id, orientation(direction).opposite());
        }
        boundaries.push_back(std::move(boundaries_for_direction));
      }
    }

    // Create NonconformingN[+/-] regions from any purely-DG boundaries,
    // using the same merging logic as NonconformingEqualRateRegions.
    if (boundaries.empty()) {
      continue;
    }

    if (boundaries.size() == 2 and
        boundaries[0].front().second.dimension() ==
            boundaries[1].front().second.dimension() and
        gsl::at(gsl::at(initial_levels, block.id()),
                boundaries[0].front().second.dimension()) > 0) {
      // Two nonconforming sides along the same axis with interior elements
      // between them: create two separate regions.
      for (auto& boundary : boundaries) {
        nonconforming_region_names_.push_back(
            "Nonconforming" + std::to_string(block.id()) +
            (boundary.front().second.side() == Side::Upper ? "+" : "-"));
        nonconforming_regions_.push_back(std::move(boundary));
      }
    } else {
      // One nonconforming side, or two sides whose outermost elements overlap:
      // merge into a single region.
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
      nonconforming_region_names_.push_back("Nonconforming" +
                                            std::to_string(block.id()));
      nonconforming_regions_.push_back(std::move(boundary));
    }
  }

  // Post-processing: resolve element-level conflicts between the Subcell region
  // and any NonconformingN regions, merging only when necessary.
  //
  // A DG-only block B may appear in both subcell_adjacent_dg_faces_ (via
  // subcell-adjacent face F1) and a NonconformingN region (via purely-DG face
  // F2). A conflict exists only if some element of B is simultaneously
  // extremal on F2 and in the Subcell region. The separability condition for
  // block B (checked against every F1 of B in subcell_adjacent_dg_faces_):
  //
  //   F1.dimension() == F2.dimension()  AND
  //   initial_refinement[B][F2.dimension()] > 0
  //
  // When this holds for all overlapping blocks, the F2-extremal elements are
  // distinct from all F1-extremal elements: the NonconformingN region is kept
  // as an independent equal-rate region, free to step at its own LTS rate.
  //
  // If ANY overlapping block fails the condition (faces on different axes, or
  // refinement 0 meaning a single element is extremal on all its faces), the
  // entire region is merged into Subcell. Merging may expose further overlaps
  // in other regions, so the loop repeats until stable.
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = nonconforming_regions_.size(); i > 0; --i) {
      const size_t idx = i - 1;

      // Determine whether a merge is required for this region.
      bool requires_merge = false;
      for (const auto& region_face : nonconforming_regions_[idx]) {
        const size_t bid = region_face.first;
        for (const auto& adj_face : subcell_adjacent_dg_faces_) {
          if (adj_face.first != bid) {
            continue;
          }
          // An element extremal on region_face could also be in Subcell if the
          // adj_face is on a different axis, or if refinement is 0 (single
          // element is extremal on all its faces simultaneously).
          if (adj_face.second.dimension() != region_face.second.dimension() or
              gsl::at(gsl::at(initial_levels, bid),
                      region_face.second.dimension()) == 0) {
            requires_merge = true;
            break;
          }
        }
        if (requires_merge) {
          break;
        }
      }

      if (not requires_merge) {
        continue;
      }

      // Merge all DG-only faces from this region into
      // subcell_adjacent_dg_faces_.  No face can already be present: a face
      // classified as subcell-adjacent in the main loop cannot appear in any
      // NonconformingN region, and each (block, direction) pair is emitted at
      // most once by the main loop so no face appears in two regions.
      for (const auto& region_face : nonconforming_regions_[idx]) {
        if (not alg::found(only_dg_block_ids_, region_face.first)) {
          continue;  // subcell-capable block, already in Subcell
        }
        ASSERT(not alg::found(subcell_adjacent_dg_faces_, region_face),
               "Face (" << region_face.first << ", " << region_face.second
                        << ") is already in subcell_adjacent_dg_faces_ before "
                           "merge; this indicates a bug in the classification "
                           "logic.");
        subcell_adjacent_dg_faces_.push_back(region_face);
      }
      nonconforming_region_names_.erase(nonconforming_region_names_.begin() +
                                        static_cast<std::ptrdiff_t>(idx));
      nonconforming_regions_.erase(nonconforming_regions_.begin() +
                                   static_cast<std::ptrdiff_t>(idx));
      changed = true;
    }
  }
}

template <size_t Dim>
std::unordered_map<std::string, size_t>
SubcellAndNonconformingEqualRateRegions<Dim>::regions() const {
  std::unordered_map<std::string, size_t> result{};
  result.emplace("Subcell", 0);
  for (size_t i = 0; i < nonconforming_region_names_.size(); ++i) {
    result.emplace(nonconforming_region_names_[i], i + 1);
  }
  return result;
}

template <size_t Dim>
bool SubcellAndNonconformingEqualRateRegions<Dim>::is_in_region(
    const size_t region, const ElementId<Dim>& element_id) const {
  if (region == 0) {
    // Subcell region: includes all subcell-capable elements, plus the outermost
    // elements of DG-only blocks that touch a subcell-adjacent nonconforming
    // face.
    if (not alg::found(only_dg_block_ids_, element_id.block_id())) {
      return true;
    }
    return alg::any_of(subcell_adjacent_dg_faces_, [&](const auto& face) {
      const auto& [block_id, direction] = face;
      if (element_id.block_id() != block_id) {
        return false;
      }
      const auto& seg = element_id.segment_id(direction.dimension());
      return seg.index() == (direction.side() == Side::Lower
                                 ? 0
                                 : two_to_the(seg.refinement_level()) - 1);
    });
  }
  // Purely DG nonconforming region.
  ASSERT(region - 1 < nonconforming_regions_.size(),
         "Region " << region << " is out of range (have "
                   << nonconforming_regions_.size()
                   << " purely-DG nonconforming regions)");
  return alg::any_of(nonconforming_regions_[region - 1], [&](const auto& face) {
    const auto& [block_id, direction] = face;
    if (element_id.block_id() != block_id) {
      return false;
    }
    const auto& seg = element_id.segment_id(direction.dimension());
    return seg.index() == (direction.side() == Side::Lower
                               ? 0
                               : two_to_the(seg.refinement_level()) - 1);
  });
}

template <size_t Dim>
void SubcellAndNonconformingEqualRateRegions<Dim>::pup(PUP::er& p) {
  p | only_dg_block_ids_;
  p | subcell_adjacent_dg_faces_;
  p | nonconforming_region_names_;
  p | nonconforming_regions_;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template class SubcellAndNonconformingEqualRateRegions<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg::subcell

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                      \
  template class evolution::dg::EqualRateRegions< \
      DIM(data),                                  \
      tmpl::list<evolution::dg::subcell::         \
                     SubcellAndNonconformingEqualRateRegions<DIM(data)>>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
