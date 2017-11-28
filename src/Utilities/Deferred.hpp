// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Deferred and make function

#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

template <typename Rt>
class Deferred;

namespace Deferred_detail {
template <typename T>
decltype(auto) retrieve_from_deferred(const T& t) {
  return t;
}

template <typename T>
decltype(auto) retrieve_from_deferred(const Deferred<T>& t) {
  return t.get();
}

template <typename T>
struct remove_deferred {
  using type = T;
};

template <typename T>
struct remove_deferred<Deferred<T>> {
  using type = T;
};

template <typename T>
using remove_deferred_t = typename remove_deferred<T>::type;

template <typename Rt>
class assoc_state {
 public:
  assoc_state() = default;
  assoc_state(const assoc_state& /*rhs*/) = delete;
  assoc_state& operator=(const assoc_state& /*rhs*/) = delete;
  assoc_state(assoc_state&& /*rhs*/) = delete;
  assoc_state& operator=(assoc_state&& /*rhs*/) = delete;
  virtual const Rt& get() const = 0;
  virtual Rt& mutate() = 0;
  virtual ~assoc_state() = default;
};

template <typename Rt>
class simple_assoc_state : public assoc_state<Rt> {
 public:
  explicit simple_assoc_state(Rt t) : t_(std::move(t)) {}

  const Rt& get() const override { return t_; }

  Rt& mutate() override { return t_; }

 private:
  Rt t_;
};

template <typename Rt, typename Fp, typename... Args>
class deferred_assoc_state : public assoc_state<Rt> {
 public:
  explicit deferred_assoc_state(Fp f, Args... args)
      : func_(std::move(f)), args_(std::make_tuple(std::move(args)...)) {}
  deferred_assoc_state(const deferred_assoc_state& /*rhs*/) = delete;
  deferred_assoc_state& operator=(const deferred_assoc_state& /*rhs*/) = delete;
  deferred_assoc_state(deferred_assoc_state&& /*rhs*/) = delete;
  deferred_assoc_state& operator=(deferred_assoc_state&& /*rhs*/) = delete;
  ~deferred_assoc_state() override = default;

  const Rt& get() const override {
    if (not evaluated_) {
      apply(std::make_index_sequence<sizeof...(Args)>{});
      evaluated_ = true;
    }
    return t_;
  }

  Rt& mutate() override {
    ERROR("Cannot mutate a computed Deferred");
    return t_;
  }

  void update_args(std::decay_t<Args>... args) noexcept {
    evaluated_ = false;
    args_ = std::tuple<std::decay_t<Args>...>{std::move(args)...};
  }

 private:
  const Fp func_;
  std::tuple<std::decay_t<Args>...> args_;
  mutable bool evaluated_ = false;
  mutable Rt t_;

  template <
      size_t... Is,
      Requires<((void)sizeof...(Is),
                tt::is_callable_v<std::decay_t<Fp>,
                                  remove_deferred_t<std::decay_t<Args>>...>)> =
          nullptr>
  void apply(std::integer_sequence<size_t, Is...> /*meta*/) const {
    t_ = std::move(func_(retrieve_from_deferred(std::get<Is>(args_))...));
  }

  template <
      size_t... Is,
      Requires<((void)sizeof...(Is),
                tt::is_callable_v<
                    std::decay_t<Fp>, gsl::not_null<std::add_pointer_t<Rt>>,
                    remove_deferred_t<std::decay_t<Args>>...>)> = nullptr>
  void apply(std::integer_sequence<size_t, Is...> /*meta*/) const {
    func_(make_not_null(&t_), retrieve_from_deferred(std::get<Is>(args_))...);
  }
};
}  // namespace Deferred_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Provides deferred or lazy evaluation of a function or function object,
 * as well as efficient storage of an object that is mutable.
 *
 * The class is similar to a std::shared_future that is able to hold and allow
 * mutation of objects. std::shared_future allows lazy evaluation of functions
 * but does not allow mutation of stored objects. Since mutation is only defined
 * for storage of objects, not for lazily evaluated functions, attempts to
 * mutate a lazily evaluated function's data is an error.
 *
 * To construct a Deferred for lazy evaluation use the make_deferred() function.
 *
 * \example
 * Construction of a Deferred with an object followed by mutation:
 * \snippet Test_Deferred.cpp deferred_with_update
 * @tparam Rt the type being stored
 */
template <typename Rt>
class Deferred {
 public:
  Deferred() = default;
  explicit Deferred(Rt t)
      : state_(std::make_shared<Deferred_detail::simple_assoc_state<Rt>>(
            std::move(t))) {}

  constexpr const Rt& get() const { return state_->get(); }

  constexpr Rt& mutate() { return state_->mutate(); }

 private:
  std::shared_ptr<Deferred_detail::assoc_state<Rt>> state_;

  explicit Deferred(std::shared_ptr<Deferred_detail::assoc_state<Rt>>&& state)
      : state_(std::move(state)) {}

  // clang-tidy: redundant declaration
  template <typename Rt1, typename Fp, typename... Args>
  friend Deferred<Rt1> make_deferred(Fp f, Args&&... args);  // NOLINT

  // clang-tidy: redundant declaration
  template <typename Rt1, typename Fp, typename... Args>
  friend void update_deferred_args(  // NOLINT
      gsl::not_null<Deferred<Rt1>*> deferred, Fp /*f used for type deduction*/,
      Args&&... args);

  // clang-tidy: redundant declaration
  template <typename Rt1, typename Fp, typename... Args>
  friend void update_deferred_args(  // NOLINT
      gsl::not_null<Deferred<Rt1>*> deferred, Args&&... args);
};

/*!
 * \ingroup UtilitiesGroup
 * \brief Create a deferred function call object
 *
 * If creating a Deferred with a function object the call operator of the
 * function object must be marked `const` currently. Since the function object
 * will only be evaluated once there currently seems to be no reason to allow
 * mutating call operators.
 *
 * \example
 * The examples below use the following functions:
 * \snippet Test_Deferred.cpp functions_used
 * To create a Deferred using a function object use:
 * \snippet Test_Deferred.cpp make_deferred_with_function_object
 * or using a regular function:
 * \snippet Test_Deferred.cpp make_deferred_with_function
 *
 * It is also possible to pass Deferred objects to a deferred function call:
 * \snippet Test_Deferred.cpp make_deferred_with_deferred_arg
 * in which case the first function will be evaluated just before the second
 * function is evaluated.
 *
 * In addition to functions that return by value, it is also possible to use
 * functions that return by reference. The first argument of the function must
 * then be a `gsl::not_null<Rt*>`, and can be mutated inside the function. The
 * mutating functions are primarily useful if `Rt` performs heap allocations and
 * is frequently recomputed in a manner where the heap allocation could be
 * avoided.
 *
 * \tparam Rt the type of the object returned by the function
 * @return Deferred object that will lazily evaluate the function
 */
template <typename Rt, typename Fp, typename... Args>
Deferred<Rt> make_deferred(Fp f, Args&&... args) {
  return Deferred<Rt>(std::make_shared<Deferred_detail::deferred_assoc_state<
                          Rt, std::decay_t<Fp>, std::decay_t<Args>...>>(
      f, std::forward<Args>(args)...));
}

// @{
/*!
 * \ingroup UtilitiesGroup
 * \brief Change the arguments to the Deferred function
 *
 * In order to make mutating Deferred functions really powerful, the `args` to
 * them must be updated without destructing the held `Rt` object. The type of
 * `Fp` (the invokable being lazily evaluated) as well as the types of the
 * `std::decay_t<Args>...` must match their respective types at the time of
 * creation of the Deferred object.
 *
 * \example
 * You can avoid specifying the type of the function held by the Deferred class
 * by passing the function as a second argument:
 * \snippet Test_Deferred.cpp update_args_of_deferred_deduced_fp
 *
 * You can also specify the type of the function held by the Deferred explicitly
 * as follows:
 * \snippet Test_Deferred.cpp update_args_of_deferred_specified_fp
 */
template <typename Rt, typename Fp, typename... Args>
void update_deferred_args(const gsl::not_null<Deferred<Rt>*> deferred,
                          Fp /*f used for type deduction*/, Args&&... args) {
  update_deferred_args<Rt, Fp>(deferred, std::forward<Args>(args)...);
}

template <typename Rt, typename Fp, typename... Args>
void update_deferred_args(const gsl::not_null<Deferred<Rt>*> deferred,
                          Args&&... args) {
  auto* ptr = dynamic_cast<Deferred_detail::deferred_assoc_state<
      Rt, std::decay_t<Fp>, std::decay_t<Args>...>*>(deferred->state_.get());
  if (ptr == nullptr) {
    ERROR("Cannot cast the Deferred class to: "s
          << (pretty_type::get_name<Deferred_detail::deferred_assoc_state<
                  Rt, std::decay_t<Fp>, std::decay_t<Args>...>>())
          << " which means you are either passing in args of incorrect "
             "types, that you are attempting to modify the args of a "
             "Deferred that is not a lazily evaluated function, or that the "
             "function type that the Deferred is expected to be holding is "
             "incorrect."s);
  }
  ptr->update_args(std::forward<Args>(args)...);
}
// @}
