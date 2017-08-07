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
      : func_(f), args_(std::make_tuple(std::move(args)...)) {}
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

 private:
  const Fp func_;
  std::tuple<std::decay_t<Args>...> args_;
  mutable bool evaluated_ = false;
  mutable Rt t_;

  template <size_t... Is>
  void apply(std::integer_sequence<size_t, Is...> /*meta*/) const {
    t_ = std::move(func_(retrieve_from_deferred(std::get<Is>(args_))...));
  }
};

template <typename T>
struct get_type_from_deferred_impl {
  using type = T;
};

template <typename T>
struct get_type_from_deferred_impl<Deferred<T>> {
  using type = T;
};

template <typename T>
using get_type_from_deferred = typename get_type_from_deferred_impl<T>::type;
}  // namespace Deferred_detail

/*!
 * \ingroup Utilities
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
  template <typename Fp, typename... Args, typename Rt1>
  friend Deferred<Rt1> make_deferred(Fp f, Args&&... args);  // NOLINT
};

/*!
 * \ingroup Utilities
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
 * @return Deferred object that will lazily evaluate the function
 */
template <typename Fp, typename... Args,
          typename Rt1 = std::result_of_t<Fp(
              Deferred_detail::get_type_from_deferred<std::decay_t<Args>>...)>>
Deferred<Rt1> make_deferred(Fp f, Args&&... args) {
  return Deferred<Rt1>(
      std::make_shared<Deferred_detail::deferred_assoc_state<
          std::decay_t<Rt1>, std::decay_t<Fp>, std::decay_t<Args>...>>(
          f, std::forward<Args>(args)...));
}
