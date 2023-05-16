// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functions for serializing factory-created classes

#pragma once

#include <boost/algorithm/string.hpp>
#include <pup.h>
#include <typeinfo>

#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \brief String representation of a type that is somewhat stable
 *
 * This string representation is used to identify the type in serialized data,
 * including data written to disk. Therefore, the registration name must be
 * somewhat reliable across invocations of the same program, across different
 * compilers, and across different executables. C++ provides no standard tool
 * for this. In particular, the name returned by `typeid().name()` explicitly
 * provides no guarantees, though for GCC and Clang it returns the mangled name
 * which should be ABI-stable. See docs:
 * https://en.cppreference.com/w/cpp/types/type_info/name
 *
 * To obtain a somewhat reliable string representation we use
 * `pretty_type::get_name` (which actually demangles the `typeid().name()`). It
 * also doesn't make any guarantees, but at least it gives a human-readable
 * string representation close to the actual source code. We remove whitespaces
 * from the name because we found inconsistent whitespace on different machines,
 * likely because the compiler's internal demangling routine got updated:
 * https://github.com/sxs-collaboration/spectre/issues/4944
 *
 * To improve reliability further in the future we may call a dedicated function
 * on `T`, such as `static std::string registration_name()`, which is expected
 * to return a unique name (including any template parameters etc). However,
 * adding this requirement to all PUP::able classes requires quite a lot of code
 * on our part (but is ultimately the safest option).
 *
 * \warning Renaming classes or namespaces will change the registration name and
 * hence break compatibility with data written by older versions of the code, as
 * does changing the implementation of this function. This means we can't
 * deserialize coordinate maps written in H5 files for interpolation of volume
 * data.
 */
template <typename T>
std::string registration_name() {
  std::string result = pretty_type::get_name<T>();
  boost::algorithm::erase_all(result, " ");
  return result;
}

/// Register specified classes.  This function can either take classes
/// to register as template arguments or take a `tmpl::list` of
/// classes as a function argument.
template <typename... Registrants>
void register_classes_with_charm(
    const tmpl::list<Registrants...> /*meta*/ = {}) {
  const auto helper = [](auto class_v) {
    using class_to_register = typename decltype(class_v)::type;
    // We use PUPable_reg2 because this takes as a second argument the name of
    // the class (as a `const char*`), while PUPable_reg converts the argument
    // verbatim to a string using the `#` preprocessor operator.
    PUPable_reg2(class_to_register,
                 registration_name<class_to_register>().c_str());
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
