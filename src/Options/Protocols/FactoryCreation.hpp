// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace Options::protocols {
/*!
 * \brief Compile-time information for factory-creation
 *
 * A class conforming to this protocol is placed in the metavariables
 * to provide information on the set of derived classes for each
 * factory-creatable base class.  The conforming class must provide
 * the following type aliases:
 *
 * - `factory_classes`: A `tmpl::map` from the base class to the list
 *   of derived classes. List all derived classes that should be
 *   factory-creatable, e.g. through input-file options.
 *
 * Here's an example for a class conforming to this protocol:
 *
 * \snippet Test_Factory.cpp factory_creation
 */
struct FactoryCreation {
  template <typename ConformingType>
  struct test {
    using factory_classes = typename ConformingType::factory_classes;

    template <typename BaseClass, typename DerivedClass>
    struct factory_class_is_derived_from_base_class : std::true_type {
      static_assert(std::is_base_of_v<BaseClass, DerivedClass>,
                    "FactoryCreation protocol: The first template argument to "
                    "this struct is not a base class of the second.");
    };

    using all_classes_are_derived_classes = tmpl::all<
        factory_classes,
        tmpl::bind<
            tmpl::all, tmpl::bind<tmpl::back, tmpl::_1>,
            tmpl::defer<factory_class_is_derived_from_base_class<
                tmpl::bind<tmpl::front, tmpl::parent<tmpl::_1>>, tmpl::_1>>>>;

    static_assert(all_classes_are_derived_classes::value);
  };
};
}  // namespace Options::protocols
