// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/Creators/OptionTags.hpp"
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
/// Generator for `EqualRateRegions` labeling all elements that are
/// allowed to do subcell and their neighbors.  The inverse of the
/// `OnlyDgBlocksAndGroups` input file option.
template <size_t Dim>
class SubcellEqualRateRegion {
 public:
  SubcellEqualRateRegion() = default;

  using creation_tags = tmpl::list<OptionTags::SubcellOptions,
                                   domain::OptionTags::DomainCreator<Dim>>;

  SubcellEqualRateRegion(
      const SubcellOptions& subcell_options,
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator);

  std::unordered_map<std::string, size_t> regions() const;

  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);

 private:
  std::vector<size_t> only_dg_block_ids_{};
};
}  // namespace evolution::dg::subcell
