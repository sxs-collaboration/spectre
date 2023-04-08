// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functions for serializing factory-created classes

#pragma once

#include <pup.h>
#include <typeinfo>

#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// Register specified classes.  This function can either take classes
/// to register as template arguments or take a `tmpl::list` of
/// classes as a function argument.
template <typename... Registrants>
void register_classes_with_charm(
    const tmpl::list<Registrants...> /*meta*/ = {}) {
  const auto helper = [](auto class_v) {
    using class_to_register = typename decltype(class_v)::type;
    // Notes:
    // - We use PUPable_reg2 because this takes as a second argument the name of
    //   the class (as a `const char*`), while PUPable_reg converts the argument
    //   verbatim to a string using the `#` preprocessor operator.
    // - We use pretty_type to get a string representation of the class that is
    //   somewhat reliable: across invocations of the same program, across
    //   different compilers, and across different executables. This reliability
    //   is important when we serialize PUPable classes to disk and read them
    //   back in. Note that typeid().name() provides no such guarantees. To
    //   improve reliability further in the future (e.g. to guard against
    //   changing pretty_type output) we may call a dedicated function on
    //   `class_to_register`, such as `static std::string registration_name()`,
    //   which is expected to return a unique name (including any template
    //   parameters etc).
    PUPable_reg2(class_to_register,
                 pretty_type::get_name<class_to_register>().c_str());
  };
  (void)helper;
  EXPAND_PACK_LEFT_TO_RIGHT(helper(tmpl::type_<Registrants>{}));
}

/// Register derived classes of the `Base` class
template <typename Base>
void register_derived_classes_with_charm() {
  register_classes_with_charm(typename Base::creatable_classes{});
}

/// Register all classes in Metavariables::factory_classes
template <typename Metavariables>
void register_factory_classes_with_charm() {
  register_classes_with_charm(
      tmpl::filter<
          tmpl::flatten<tmpl::values_as_sequence<
              typename Metavariables::factory_creation::factory_classes>>,
          std::is_base_of<PUP::able, tmpl::_1>>{});
}
