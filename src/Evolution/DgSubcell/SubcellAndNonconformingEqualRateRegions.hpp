// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class DomainCreator;
template <size_t VolumeDim>
class ElementId;
namespace PUP {
class er;
}  // namespace PUP
namespace evolution::dg::subcell {
class SubcellOptions;
}  // namespace evolution::dg::subcell
/// \endcond

namespace evolution::dg::subcell {
/// Combined equal-rate region generator for subcell evolution with
/// nonconforming block boundaries.
///
/// When using local time stepping (LTS) with both subcell evolution and
/// nonconforming block boundaries, the two separate region generators
/// `NonconformingEqualRateRegions` and `SubcellEqualRateRegion`
/// cannot be combined without conflicts in their regions.  This class merges
/// the two generators into one, treating subcell-adjacent nonconforming
/// interfaces as part of the "Subcell" region rather than as separate
/// "NonconformingN" regions.
///
/// **Subcell region (region 0)**: Contains all subcell-capable elements (those
/// not in `OnlyDgBlocksAndGroups`) **plus** the elements of DG-only
/// blocks that lie directly on a nonconforming face bordering a subcell-capable
/// block.
///
/// **NonconformingN[+/-] regions (regions 1…N)**: Created only for
/// nonconforming interfaces where every block on both sides is DG-only (no
/// subcell capability on either side).  These behave identically to the
/// regions produced by `NonconformingEqualRateRegions`.
///
/// \note If your domain has no nonconforming block boundaries, use
/// `SubcellEqualRateRegion` instead: it is simpler and has no overhead.
template <size_t Dim>
class SubcellAndNonconformingEqualRateRegions {
 public:
  SubcellAndNonconformingEqualRateRegions() = default;

  using creation_tags = tmpl::list<OptionTags::SubcellOptions,
                                   domain::OptionTags::DomainCreator<Dim>>;

  SubcellAndNonconformingEqualRateRegions(
      const SubcellOptions& subcell_options,
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator);

  std::unordered_map<std::string, size_t> regions() const;

  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);

 private:
  // DG-only block IDs: non-hypercube topology blocks (detected automatically)
  // plus user-specified OnlyDgBlocksAndGroups blocks. Subcell-capable blocks
  // (those NOT in this list) are always in region 0 ("Subcell")
  std::vector<size_t> only_dg_block_ids_{};
  // (block_id, direction) pairs identifying the faces of DG-only blocks that
  // touch a nonconforming boundary with at least one subcell-capable neighbor.
  // The elements on these faces are included in region 0 ("Subcell").
  std::vector<std::pair<size_t, Direction<Dim>>> subcell_adjacent_dg_faces_{};
  // Names and face-membership data for purely DG nonconforming regions
  // (region index i here corresponds to region i+1 overall, since region 0
  // is "Subcell").
  std::vector<std::string> nonconforming_region_names_{};
  std::vector<std::vector<std::pair<size_t, Direction<Dim>>>>
      nonconforming_regions_{};
};
}  // namespace evolution::dg::subcell
