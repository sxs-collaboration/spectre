// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>
#include <typeinfo>

#include "ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup Utilities
/// \brief Define a function that acts similarly to a virtual
/// function, but can take template parameters.
///
/// \details `DEFINE_FAKE_VIRTUAL(func)` defines the function
/// `fake_virtual_func` and the struct `FakeVirtualInherit_func`.  It
/// should usually be called in a detail namespace.
///
/// A base class `Base` using this functionality should define a type
/// \code
/// using Inherit = FakeVirtualInherit_func<Base>;
/// \endcode
/// and a member function `func` wrapping `fake_virtual_func`, with
/// the wrapper passing the derived classes as a typelist as the first
/// template argument and the `this` pointer as the first normal
/// argument.
///
/// Derived classes should then inherit from `Base::Inherit` instead
/// of directly from `Base`.  (`Base::Inherit` inherits from `Base`.)
///
/// If the base class has no pure virtual functions remaining it will
/// generally be desirable to mark the constructors and assignment
/// operators protected so that a bare base class cannot be instantiated.
///
/// If it is necessary to use multiple fake virtual functions with the
/// same base class, the `Inherit` definition can nest the fake
/// virtual classes:
/// \code
/// using Inherit = FakeVirtualInherit_func1<FakeVirtualInherit_func2<Base>>;
/// \endcode
///
/// \example
/// \snippet FakeVirtual.cpp fake_virtual_example
#define DEFINE_FAKE_VIRTUAL(function)                                          \
  /* This struct is only needed for producing an error if the function */      \
  /* is not overridden in the derived class. */                                \
  template <typename Base>                                                     \
  struct FakeVirtualInherit_##function : public Base {                         \
    using Base::Base;                                                          \
    void function(...) const = delete;                                         \
  };                                                                           \
                                                                               \
  template <                                                                   \
      typename Classes, typename... TArgs, typename Base, typename... Args,    \
      typename std::enable_if_t<(tmpl::size<Classes>::value == 0)>* = nullptr> \
  auto fake_virtual_##function(Base* obj, Args&&... args)                      \
      ->decltype(obj->template function<TArgs...>(args...)) {                  \
    ERROR("Class " << pretty_type::get_runtime_type_name(*obj)                 \
                   << " is not registered with "                               \
                   << pretty_type::get_name<std::remove_const_t<Base>>());     \
  }                                                                            \
                                                                               \
  template <                                                                   \
      typename Classes, typename... TArgs, typename Base, typename... Args,    \
      typename std::enable_if_t<(tmpl::size<Classes>::value != 0)>* = nullptr> \
  decltype(auto) fake_virtual_##function(Base* obj, Args&&... args) {          \
    using derived = tmpl::front<Classes>;                                      \
    static_assert(std::is_base_of<typename Base::Inherit, derived>::value,     \
                  "Derived class does not inherit from Base::Inherit");        \
    using derived_p = std::conditional_t<std::is_const<Base>::value,           \
                                         derived const*, derived*>;            \
    /* If we want to allow creatable classses to return objects of */          \
    /* types derived from themselves then this will have to be changed */      \
    /* to a dynamic_cast, but we probably won't want that and this */          \
    /* form is significantly faster. */                                        \
    if (typeid(*obj) == typeid(derived)) {                                     \
      return static_cast<derived_p>(obj)->template function<TArgs...>(         \
          std::forward<Args>(args)...);                                        \
    } else {                                                                   \
      return fake_virtual_##function<tmpl::pop_front<Classes>, TArgs...>(      \
          obj, std::forward<Args>(args)...);                                   \
    }                                                                          \
  }
