// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

/// \ingroup UtilitiesGroup
/// \brief Helpers for derived class registration
///
/// SpECTRE's factory mechanism requires that the base class contain a
/// list of all derived classes (in the special nested type alias
/// `creatable_classes`).  This can be problematic when the desired
/// list of derived classes can vary with use of the base class.
/// These helpers provide a method to simplify handling these
/// compile-time registrations.
///
/// Each optional derived class defines a registrar helper and both
/// the base class and derived classes are templated on the list of
/// registrars.  The typical structure of these classes is:
/// \snippet Test_Registration.cpp registrar_structure
/// A concrete base class type with a specific set of derived classes
/// is constructed like
/// \snippet Test_Registration.cpp registrar_use
///
/// It is frequently useful to default the registrar list in a derived
/// class to the registrar for that class.  This ensures that any
/// methods in the base class using the registrar list (such as those
/// using `DEFINE_FAKE_VIRTUAL()` or `call_with_dynamic_type()`) work
/// as expected on an explicitly constructed derived class with the
/// list omitted.
namespace Registration {
/// A template for defining a registrar.
///
/// A registrar for a class can be defined by making a type alias to
/// this struct, filling in the registrant.
/// \snippet Test_Registration.cpp registrar
/// In more complex cases (such as with non-type template
/// parameters) defining a registrar manually may be necessary.
/// \snippet Test_Registration.cpp custom_registrar
template <template <typename...> class Registrant, typename... Args>
struct Registrar {
  // Final registrant type, specialized with input Args... and the
  // RegistrarList it was extracted from.  This will be a full
  // specialization of a derived class that uses the registration
  // framework.
  template <typename RegistrarList>
  using f = Registrant<Args..., RegistrarList>;
};

namespace detail {
template <typename RegistrarList, typename Registrar>
struct registrant {
  using type = typename Registrar::template f<RegistrarList>;
};
}  // namespace detail

/// Transform a list of registrars into the list of associated
/// registrants.  This is usually used to define the
/// `creatable_classes` type list.
template <typename RegistrarList>
using registrants = tmpl::transform<
    tmpl::remove_duplicates<RegistrarList>,
    detail::registrant<tmpl::pin<tmpl::remove_duplicates<RegistrarList>>,
                       tmpl::_1>>;
}  // namespace Registration
