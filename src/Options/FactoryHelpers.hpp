// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

namespace Options {
namespace detail {
template <typename FactoryClasses, typename... NewClasses>
struct add_factory_classes;

template <typename FactoryClasses, typename BaseClass, typename NewClasses,
          typename... RestNewClasses>
struct add_factory_classes<FactoryClasses, tmpl::pair<BaseClass, NewClasses>,
                           RestNewClasses...>
    : add_factory_classes<
          tmpl::insert<
              tmpl::erase<FactoryClasses, BaseClass>,
              tmpl::pair<
                  BaseClass,
                  tmpl::append<
                      tmpl::conditional_t<
                          tmpl::has_key<FactoryClasses, BaseClass>::value,
                          tmpl::at<FactoryClasses, BaseClass>, tmpl::list<>>,
                      NewClasses>>>,
          RestNewClasses...> {};

template <typename FactoryClasses>
struct add_factory_classes<FactoryClasses> {
  using type = FactoryClasses;
};
}  // namespace detail

/// Add new factory-creatable classes to the list in a
/// `factory_creation` struct.
///
/// For each `tmpl::pair<Base, DerivedList>` in the `NewClasses`
/// parameter pack, append the classes in `DerivedList` to the list
/// for `Base` in `FactoryClasses`.  The `FactoryClasses` map need not
/// have a preexisting entry for `Base`.
///
/// This example adds three new derived classes, two for one base
/// class and one for another.
///
/// \snippet Test_FactoryHelpers.cpp add_factory_classes
///
/// \see Options::protocols::FactoryCreation
template <typename FactoryClasses, typename... NewClasses>
using add_factory_classes =
    typename detail::add_factory_classes<FactoryClasses, NewClasses...>::type;
}  // namespace Options
