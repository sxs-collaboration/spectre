// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Imex/Tags/Jacobian.hpp"
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
/// Protocol for the Jacobian of implicit source terms.
///
/// A struct conforming to this protocol must also conform to
/// ::protocols::StaticReturnApplyable.  The `return_tags` of this
/// struct must be a list of `Tags::Jacobian` tags for the dependence
/// of the sector's `Tags::Sources` on the sector's tensors.  Portions
/// of the Jacobian that are zero may be omitted from the list.
///
/// The `argument_tags` must not include compute items depending on
/// tensors in the implicit sector.
///
/// \snippet Test_SolveImplicitSector.cpp Jacobian
struct ImplicitSourceJacobian {
  template <typename ConformingType>
  struct test {
    static_assert(tt::assert_conforms_to_v<ConformingType,
                                           ::protocols::StaticReturnApplyable>);

    using return_tags = typename ConformingType::return_tags;
    static_assert(
        tmpl::all<return_tags, tt::is_a<Tags::Jacobian, tmpl::_1>>::value);

    template <typename T>
    struct get_dependent {
      using type = typename T::dependent;
    };

    static_assert(
        tmpl::all<return_tags,
                  tt::is_a<::Tags::Source, get_dependent<tmpl::_1>>>::value);
  };
};
}  // namespace imex::protocols
