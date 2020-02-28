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
/// Register the classes in `ListOfClassesToRegister` with the serialization
/// framework.
template <typename ListOfClassesToRegister>
void register_classes_in_list() noexcept {
  tmpl::for_each<ListOfClassesToRegister>([](auto class_v) noexcept {
    using class_to_register = typename decltype(class_v)::type;
    // We use PUPable_reg2 because this takes as a second argument the name of
    // the class as a `const char*`, while PUPable_reg converts the argument
    // verbatim to a string using the `#` preprocessor operator.
    PUPable_reg2(class_to_register, typeid(class_to_register).name());
  });
}

/// Register derived classes of the `Base` class
template <typename Base>
void register_derived_classes_with_charm() noexcept {
  register_classes_in_list<typename Base::creatable_classes>();
}
}  // namespace Parallel
