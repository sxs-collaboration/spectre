// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Protocols/StaticReturnApplyable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Source;
}  // namespace Tags
/// \endcond

namespace imex::protocols {
/// Protocol for implicit source terms.
///
/// A struct conforming to this protocol must also conform to
/// ::protocols::StaticReturnApplyable.  The `return_tags` of this
/// struct must be a list of `::Tags::Source<T>` for every tensor `T`
/// in the implicit sector.
///
/// The `argument_tags` must not include compute items depending on
/// tensors in the implicit sector.
///
/// \snippet Test_SolveImplicitSector.cpp source
struct ImplicitSource {
  template <typename ConformingType>
  struct test {
    static_assert(tt::assert_conforms_to_v<ConformingType,
                                           ::protocols::StaticReturnApplyable>);

    using return_tags = typename ConformingType::return_tags;
    static_assert(
        tmpl::all<return_tags, tt::is_a<::Tags::Source, tmpl::_1>>::value,
        "All return tags must be ::Tags::Source<...>");
  };
};
}  // namespace imex::protocols
