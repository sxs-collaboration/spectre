// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>
#include <typeinfo>
#include <utility>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename Result, typename Classes, typename Base, typename Callable,
          Requires<(tmpl::size<Classes>::value == 0)> = nullptr>
[[noreturn]] Result call_with_dynamic_type(Base* const obj, Callable&& /*f*/) {
  ERROR("Class " << pretty_type::get_runtime_type_name(*obj)
        << " is not registered with "
        << pretty_type::get_name<std::remove_const_t<Base>>());
}
/// \endcond

/// \ingroup UtilitiesGroup
/// \brief Call a functor with the derived type of a base class pointer.
///
/// \details Calls functor with obj cast to type `T*` where T is the
/// dynamic type of `*obj`.  The decay type of `T` must be in the
/// provided list of classes.
///
/// \see DEFINE_FAKE_VIRTUAL
///
/// \tparam Result the return type
/// \tparam Classes the typelist of derived classes
template <typename Result, typename Classes, typename Base, typename Callable,
          Requires<(tmpl::size<Classes>::value != 0)> = nullptr>
Result call_with_dynamic_type(Base* const obj, Callable&& f) {
  using Derived = tmpl::front<Classes>;
  using DerivedPointer =
      tmpl::conditional_t<std::is_const<Base>::value, Derived const*, Derived*>;
  // If we want to allow creatable classses to return objects of
  // types derived from themselves then this will have to be changed
  // to a dynamic_cast, but we probably won't want that and this
  // form is significantly faster.
  return typeid(*obj) == typeid(Derived)
             ? std::forward<Callable>(f)(static_cast<DerivedPointer>(obj))
             : call_with_dynamic_type<Result, tmpl::pop_front<Classes>>(
                   obj, std::forward<Callable>(f));
}
