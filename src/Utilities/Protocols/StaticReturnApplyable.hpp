// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace protocols {
/// Protocol for a struct with a static `apply` function returning and
/// taking arguments based on tags in `return_tags` and
/// `argument_tags` type aliases.
///
/// \snippet Test_StaticReturnApplyable.cpp StaticReturnApplyable
struct StaticReturnApplyable {
  template <typename ConformingType>
  struct test {
    using argument_tags = typename ConformingType::argument_tags;
    using return_tags = typename ConformingType::return_tags;
    static_assert(tt::is_a_v<tmpl::list, argument_tags>);
    static_assert(tt::is_a_v<tmpl::list, return_tags>);

    static_assert(
        std::is_same_v<return_tags, tmpl::remove_duplicates<return_tags>>,
        "return_tags should not contain duplicates.");
    // Duplicates in argument_tags are OK.

    static_assert(
        tmpl::wrap<
            tmpl::append<
                tmpl::list<decltype(ConformingType::apply)>,
                tmpl::transform<
                    return_tags,
                    tmpl::bind<gsl::not_null, std::add_pointer<tmpl::bind<
                                                  tmpl::type_from, tmpl::_1>>>>,
                tmpl::transform<argument_tags,
                                std::add_lvalue_reference<std::add_const<
                                    tmpl::bind<tmpl::type_from, tmpl::_1>>>>>,
            std::is_invocable>::value,
        "apply function is not callable with the provided arguments.");
  };
};
}  // namespace protocols
