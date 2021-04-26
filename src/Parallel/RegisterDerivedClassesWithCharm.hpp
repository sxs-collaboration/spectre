// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functions for serializing factory-created classes

#pragma once

#include <pup.h>
#include <typeinfo>

#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
/// Register specified classes.  This function can either take classes
/// to register as template arguments or take a `tmpl::list` of
/// classes as a function argument.
template <typename... Registrants>
void register_classes_with_charm(
    const tmpl::list<Registrants...> /*meta*/ = {}) noexcept {
  const auto helper = [](auto class_v) noexcept {
    using class_to_register = typename decltype(class_v)::type;
    // We use PUPable_reg2 because this takes as a second argument the name of
    // the class as a `const char*`, while PUPable_reg converts the argument
    // verbatim to a string using the `#` preprocessor operator.
    PUPable_reg2(class_to_register, typeid(class_to_register).name());
  };
  (void)helper;
  EXPAND_PACK_LEFT_TO_RIGHT(helper(tmpl::type_<Registrants>{}));
}

/// Register derived classes of the `Base` class
template <typename Base>
void register_derived_classes_with_charm() noexcept {
  register_classes_with_charm(typename Base::creatable_classes{});
}
}  // namespace Parallel
