// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "Options/Options.hpp"

/// \cond
template <size_t Dim>
class DomainCreator;
/// \endcond

namespace domain {
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ComputationalDomainGroup
/// The input file tag for the DomainCreator to use
template <size_t Dim>
struct DomainCreator {
  using type = std::unique_ptr<::DomainCreator<Dim>>;
  static constexpr OptionString help = {"The domain to create initially"};
};
}  // namespace OptionTags
}  // namespace domain
