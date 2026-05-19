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

namespace evolution::dg {
/// Equal-rate region generator for regions on nonconforming block boundaries.
///
/// For each block that borders multiple other blocks on a single
/// side, defines a region "NonconformingN", where "N" is the block
/// number, consisting of all elements on the nonconforming boundary
/// in that block and the neighboring blocks.  If the block has
/// nonconforming boundaries on both sides and is more than one
/// element across, defines two regions "NonconformingN+" and
/// "NonconformingN-".
template <size_t Dim>
class NonconformingEqualRateRegions {
 public:
  NonconformingEqualRateRegions() = default;

  using creation_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>>;

  explicit NonconformingEqualRateRegions(
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator);

  std::unordered_map<std::string, size_t> regions() const;

  bool is_in_region(size_t region, const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p);

 private:
  std::vector<std::string> region_names_{};
  std::vector<std::vector<std::pair<size_t, Direction<Dim>>>> regions_{};
};
}  // namespace evolution::dg
