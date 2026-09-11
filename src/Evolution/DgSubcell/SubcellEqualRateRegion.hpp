// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/Creators/OptionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/OptionTags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class DomainCreator;
template <size_t VolumeDim>
class ElementId;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg::subcell {
/// Generator for `EqualRateRegions` labeling all elements that are
/// allowed to do subcell and their neighbors.  The inverse of the
/// `OnlyDgBlocksAndGroups` input file option.
///
/// \note When subcell is combined with nonconforming block boundaries (e.g.,
/// `domain::creators::NonconformingSphericalShells`), use
/// `SubcellAndNonconformingEqualRateRegions` instead of pairing this class
/// with `NonconformingEqualRateRegions`.
template <size_t Dim>
class SubcellEqualRateRegion {
 public:
  SubcellEqualRateRegion() = default;

  using creation_tags =
      tmpl::list<evolution::dg::OptionTags::OnlyDgBlocksAndGroups,
                 domain::OptionTags::DomainCreator<Dim>>;

  SubcellEqualRateRegion(
      const std::optional<std::vector<std::string>>&
          only_dg_block_and_group_names,
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator);

  std::unordered_map<std::string, size_t> regions() const;

  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);

 private:
  std::vector<size_t> only_dg_block_ids_{};
};
}  // namespace evolution::dg::subcell
