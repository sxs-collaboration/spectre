// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "Options/String.hpp"

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
  static constexpr Options::String help = {"The domain to create initially"};
};
}  // namespace OptionTags
}  // namespace domain
