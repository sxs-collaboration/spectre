// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>
#include <typeinfo>

#include "ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \ingroup UtilitiesGroup
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
/// \snippet Test_FakeVirtual.cpp fake_virtual_example
///
/// \see call_with_dynamic_type
#define DEFINE_FAKE_VIRTUAL(function)                                          \
  /* This struct is only needed for producing an error if the function */      \
  /* is not overridden in the derived class. */                                \
  template <typename Base>                                                     \
  struct FakeVirtualInherit_##function : public Base {                         \
    using Base::Base;                                                          \
    /* clang-tidy: I think "= delete" was overlooked in the guideline */       \
    void function(...) const = delete; /* NOLINT */                            \
  };                                                                           \
                                                                               \
  template <typename Classes, typename... TArgs, typename Base,                \
            typename... Args>                                                  \
  decltype(auto) fake_virtual_##function(Base* obj, Args&&... args) noexcept { \
    /* clang-tidy: macro arg in parentheses */                                 \
    return call_with_dynamic_type<                                             \
        decltype(obj->template function<TArgs...>(args...)), /* NOLINT */      \
        Classes>(                                                              \
        obj, [&args...](auto* const dynamic_obj) noexcept -> decltype(auto) {  \
          static_assert(                                                       \
              cpp17::is_base_of_v<typename Base::Inherit,                      \
                                  std::decay_t<decltype(*dynamic_obj)>>,       \
              "Derived class does not inherit from Base::Inherit");            \
          /* clang-tidy: macro arg in parentheses */                           \
          return dynamic_obj->template function<TArgs...>(/* NOLINT */         \
                                                          std::forward<Args>(  \
                                                              args)...);       \
        });                                                                    \
  }

/// \cond
template <typename Result, typename Classes, typename Base, typename Callable,
          Requires<(tmpl::size<Classes>::value == 0)> = nullptr>
[[noreturn]] Result call_with_dynamic_type(Base* const obj,
                                           Callable&& /*f*/) noexcept {
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
Result call_with_dynamic_type(Base* const obj, Callable&& f) noexcept {
  using Derived = tmpl::front<Classes>;
  using DerivedPointer =
      std::conditional_t<std::is_const<Base>::value, Derived const*, Derived*>;
  // If we want to allow creatable classses to return objects of
  // types derived from themselves then this will have to be changed
  // to a dynamic_cast, but we probably won't want that and this
  // form is significantly faster.
  return typeid(*obj) == typeid(Derived)
             ? std::forward<Callable>(f)(static_cast<DerivedPointer>(obj))
             : call_with_dynamic_type<Result, tmpl::pop_front<Classes>>(
                   obj, std::forward<Callable>(f));
}
