// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>
#include <typeinfo>
#include <utility>

#include "TypeTraits/FastPointerCast.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup UtilitiesGroup
/// \brief Call a functor with the derived type of a base class pointer.
///
/// \details Calls functor with obj cast to type `T*` where T is the
/// dynamic type of `*obj`.  The decay type of `T` must be in the
/// provided list of classes.
///
/// \tparam Result the return type
/// \tparam Classes the typelist of derived classes
template <typename Result, typename Classes, typename Base, typename Callable>
Result call_with_dynamic_type(Base* const obj, Callable&& f) {
  if constexpr (tmpl::size<Classes>::value != 0) {
    using Derived = tmpl::front<Classes>;
    using DerivedPointer = tmpl::conditional_t<std::is_const<Base>::value,
                                               Derived const*, Derived*>;
    return typeid(*obj) == typeid(Derived)
               ? std::forward<Callable>(f)(
                     tt::fast_pointer_cast<DerivedPointer>(obj))
               : call_with_dynamic_type<Result, tmpl::pop_front<Classes>>(
                     obj, std::forward<Callable>(f));
  } else {
    ERROR("Class " << pretty_type::get_runtime_type_name(*obj)
                   << " is not registered with "
                   << pretty_type::get_name<std::remove_const_t<Base>>());
  }
}
