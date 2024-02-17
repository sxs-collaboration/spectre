// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel::OptionTags {
struct Parallelization;
}  // namespace Parallel::OptionTags
/// \endcond

namespace domain {
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ComputationalDomainGroup
struct ElementDistribution {
  struct RoundRobin {};
  using type = Options::Auto<ElementWeight, RoundRobin>;
  static constexpr Options::String help = {
      "Weighting pattern to use for ZCurve element distribution. Specify "
      "RoundRobin to just place each element on the next core."};
  using group = Parallel::OptionTags::Parallelization;
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Tag that holds method for how to distribute the elements on the given
/// resources.
///
/// \note When not using local time stepping (LTS), a user cannot choose the
/// NumGridPointsAndGridSpacing element distribution because grid spacing does
/// not affect the computational cost at all. Therefore, if a user does choose
/// NumGridPointsAndGridSpacing when not using LTS, an error will occur.
struct ElementDistribution : db::SimpleTag {
  using type = std::optional<ElementWeight>;
  using option_tags = tmpl::list<OptionTags::ElementDistribution>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& element_distribution) {
    return element_distribution;
  }
};
}  // namespace Tags
}  // namespace domain
