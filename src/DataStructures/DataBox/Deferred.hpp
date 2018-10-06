// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Deferred and make function

#pragma once

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

template <typename Rt, typename MakeConstReference = std::false_type>
class Deferred;

namespace Deferred_detail {
template <typename T>
decltype(auto) retrieve_from_deferred(const T& t) noexcept {
  return t;
}

template <typename T, typename MakeConstReference>
decltype(auto) retrieve_from_deferred(
    const Deferred<T, MakeConstReference>& t) noexcept {
  return t.get();
}

template <typename T>
struct remove_deferred {
  using type = T;
};

template <typename T, typename MakeConstReference>
struct remove_deferred<Deferred<T, MakeConstReference>> {
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
  virtual const Rt& get() const noexcept = 0;
  virtual Rt& mutate() noexcept = 0;
  virtual void reset() noexcept = 0;
  // clang-tidy: no non-const references
  virtual void pack_unpack_lazy_function(PUP::er& p) noexcept = 0;  // NOLINT
  virtual bool evaluated() const noexcept = 0;
  virtual boost::shared_ptr<assoc_state<Rt>> deep_copy() const noexcept = 0;
  virtual ~assoc_state() = default;
};

template <typename Rt>
class simple_assoc_state : public assoc_state<Rt> {
 public:
  explicit simple_assoc_state(Rt t) noexcept;

  const Rt& get() const noexcept override { return t_; }

  Rt& mutate() noexcept override { return t_; }

  void reset() noexcept override { ERROR("Cannot reset a simple_assoc_state"); }

  // clang-tidy: no non-const references
  void pack_unpack_lazy_function(PUP::er& /*p*/) noexcept override {  // NOLINT
    ERROR("Cannot send a Deferred that's not a lazily evaluated function");
  }

  bool evaluated() const noexcept override { return true; }

  boost::shared_ptr<assoc_state<Rt>> deep_copy() const noexcept override {
    return deep_copy_impl();
  }

 private:
  template <typename T = Rt,
            Requires<tt::can_be_copy_constructed_v<T>> = nullptr>
  boost::shared_ptr<assoc_state<Rt>> deep_copy_impl() const noexcept {
    return boost::make_shared<simple_assoc_state>(t_);
  }

  template <typename T = Rt,
            Requires<not tt::can_be_copy_constructed_v<T>> = nullptr>
  boost::shared_ptr<assoc_state<Rt>> deep_copy_impl() const noexcept {
    ERROR(
        "Cannot create a copy of a DataBox (e.g. using db::create_copy) that "
        "holds a non-copyable simple item. The item type is '"
        << pretty_type::get_name<T>() << "'.");
  }

  Rt t_;
};

template <typename Rt>
simple_assoc_state<Rt>::simple_assoc_state(Rt t) noexcept : t_(std::move(t)) {}

template <typename Rt, typename Fp, typename... Args>
class deferred_assoc_state : public assoc_state<Rt> {
 public:
  explicit deferred_assoc_state(Fp f, Args... args) noexcept;
  deferred_assoc_state(const deferred_assoc_state& /*rhs*/) = delete;
  deferred_assoc_state& operator=(const deferred_assoc_state& /*rhs*/) = delete;
  deferred_assoc_state(deferred_assoc_state&& /*rhs*/) = delete;
  deferred_assoc_state& operator=(deferred_assoc_state&& /*rhs*/) = delete;
  ~deferred_assoc_state() override = default;

  const Rt& get() const noexcept override {
    if (not evaluated_) {
      apply(std::make_index_sequence<sizeof...(Args)>{});
      evaluated_ = true;
    }
    return t_;
  }

  Rt& mutate() noexcept override { ERROR("Cannot mutate a computed Deferred"); }

  void reset() noexcept override { evaluated_ = false; }

  void update_args(std::decay_t<Args>... args) noexcept {
    evaluated_ = false;
    args_ = std::tuple<std::decay_t<Args>...>{std::move(args)...};
  }

  // clang-tidy: no non-const references
  void pack_unpack_lazy_function(PUP::er& p) noexcept override {  // NOLINT
    p | evaluated_;
    if (evaluated_) {
      p | t_;
    }
  }

  bool evaluated() const noexcept override { return evaluated_; }

  boost::shared_ptr<assoc_state<Rt>> deep_copy() const noexcept override {
    ERROR(
        "Have not yet implemented a deep_copy for deferred_assoc_state. It's "
        "not at all clear if this is even possible because it is incorrect to "
        "assume that the args_ have not changed.");
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
  void apply(std::integer_sequence<size_t, Is...> /*meta*/) const noexcept {
    t_ = std::move(func_(retrieve_from_deferred(std::get<Is>(args_))...));
  }

  template <
      size_t... Is,
      Requires<((void)sizeof...(Is),
                tt::is_callable_v<
                    std::decay_t<Fp>, gsl::not_null<std::add_pointer_t<Rt>>,
                    remove_deferred_t<std::decay_t<Args>>...>)> = nullptr>
  void apply(std::integer_sequence<size_t, Is...> /*meta*/) const noexcept {
    func_(make_not_null(&t_), retrieve_from_deferred(std::get<Is>(args_))...);
  }
};

template <typename Rt, typename Fp, typename... Args>
deferred_assoc_state<Rt, Fp, Args...>::deferred_assoc_state(
    Fp f, Args... args) noexcept
    : func_(std::move(f)), args_(std::make_tuple(std::move(args)...)) {}

// Specialization to handle functions that return a `const Rt&`. We treat the
// return value as pointer to the data we actually want to have visible to us
// when we retrieve the data. The reason for using a pointer is because we need
// to be able to rebind in case the memory address of `const Rt&` changes (for
// example, if we point to a `ConstGlobalCache` and we are migrated to a
// different node). Since lvalue references cannot be rebound, we store a
// pointer. The `get` function dereferences the pointer we store so that we have
// a const lvalue reference to work with when retrieving the data being pointed
// to. Dereferencing the pointer ensures that all functions that use the DataBox
// will be able to take a `const T& t` as input argument regardless of where the
// data is stored (in the DataBox or as a reference to somewhere else).
template <typename Rt, typename Fp, typename... Args>
class deferred_assoc_state<const Rt&, Fp, Args...> : public assoc_state<Rt> {
 public:
  explicit deferred_assoc_state(Fp f, Args... args) noexcept;
  deferred_assoc_state(const deferred_assoc_state& /*rhs*/) = delete;
  deferred_assoc_state& operator=(const deferred_assoc_state& /*rhs*/) = delete;
  deferred_assoc_state(deferred_assoc_state&& /*rhs*/) = delete;
  deferred_assoc_state& operator=(deferred_assoc_state&& /*rhs*/) = delete;
  ~deferred_assoc_state() override = default;

  const Rt& get() const noexcept override {
    if (not t_) {
      apply(std::make_index_sequence<sizeof...(Args)>{});
    }
    return *t_;
  }

  Rt& mutate() noexcept override { ERROR("Cannot mutate a compute tag."); }

  void reset() noexcept override { t_ = nullptr; }

  void update_args(std::decay_t<Args>... args) noexcept {
    t_ = nullptr;
    args_ = std::tuple<std::decay_t<Args>...>{std::move(args)...};
  }

  // clang-tidy: no non-const references
  void pack_unpack_lazy_function(PUP::er& /*p*/) noexcept override {}  // NOLINT

  bool evaluated() const noexcept override { return t_ != nullptr; }

  boost::shared_ptr<assoc_state<Rt>> deep_copy() const noexcept override {
    ERROR(
        "Have not yet implemented a deep_copy for deferred_assoc_state. It's "
        "not at all clear if this is even possible because it is incorrect to "
        "assume that the args_ have not changed.");
  }

 private:
  const Fp func_;
  std::tuple<std::decay_t<Args>...> args_;
  mutable const Rt* t_ = nullptr;

  template <size_t... Is>
  void apply(std::integer_sequence<size_t, Is...> /*meta*/) const noexcept {
    t_ = &(func_(retrieve_from_deferred(std::get<Is>(args_))...));
  }
};

template <typename Rt, typename Fp, typename... Args>
deferred_assoc_state<const Rt&, Fp, Args...>::deferred_assoc_state(
    Fp f, Args... args) noexcept
    : func_(std::move(f)), args_(std::make_tuple(std::move(args)...)) {}
}  // namespace Deferred_detail

/*!
 * \ingroup DataBoxGroup
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
 *
 * \warning If passed a lazy function that returns a `const Rt& t` (a const
 * lvalue reference) the underlying data stored is actually a pointer to `t`,
 * which will be dereferenced upon retrieval. This could lead to a dangling
 * pointer/reference if care isn't taken. The reason for this design is that:
 * 1. We need to be able to retrieve data stored in the ConstGlobalCache from
 *    the DataBox. The way to do this is to have a pointer to the
 *    ConstGlobalCache inside the DataBox alongside compute items that return
 *    `const Rt&` to the ConstGlobalCache data.
 * 2. Functions used in the DataBox shouldn't return references, they are
 *    compute tags and should return by value. If we find an actual use-case for
 *    compute tags returning `const Rt&` where referencing behavior is undesired
 *    then this design needs to be reconsidered. It is currently the least
 *    breaking way to implement referencing DataBox members that can point to
 *    any memory.
 *
 * @tparam Rt the type being stored
 */
template <typename Rt, typename MakeConstReference>
class Deferred {
 public:
  using value_type = std::remove_const_t<std::remove_reference_t<Rt>>;

  Deferred() = default;
  template <typename Dummy = Rt,
            Requires<cpp17::is_same_v<Dummy, value_type>> = nullptr>
  explicit Deferred(Rt t)
      : state_(boost::make_shared<Deferred_detail::simple_assoc_state<Rt>>(
            std::move(t))) {}
  Deferred(const Deferred&) = default;
  Deferred& operator=(const Deferred&) = default;
  Deferred(Deferred&&) = default;
  Deferred& operator=(Deferred&&) = default;
  ~Deferred() = default;

  constexpr const value_type& get() const noexcept { return state_->get(); }

  constexpr value_type& mutate() noexcept { return state_->mutate(); }

  // clang-tidy: no non-const references
  void pack_unpack_lazy_function(PUP::er& p) noexcept {  // NOLINT
    state_->pack_unpack_lazy_function(p);
  }

  bool evaluated() const noexcept { return state_->evaluated(); }

  void reset() noexcept { state_->reset(); }

  Deferred deep_copy() const noexcept { return Deferred{state_->deep_copy()}; }

  explicit Deferred(
      boost::shared_ptr<Deferred_detail::assoc_state<tmpl::conditional_t<
          MakeConstReference::value, const value_type&, value_type>>>&&
          state) noexcept;

 private:
  boost::shared_ptr<Deferred_detail::assoc_state<tmpl::conditional_t<
      MakeConstReference::value, const value_type&, value_type>>>
      state_{nullptr};

  // clang-tidy: redundant declaration
  template <typename Rt1, typename Fp, typename... Args>
  friend void update_deferred_args(  // NOLINT
      gsl::not_null<Deferred<Rt1>*> deferred, Fp /*f used for type deduction*/,
      Args&&... args) noexcept;

  // clang-tidy: redundant declaration
  template <typename Rt1, typename Fp, typename... Args>
  friend void update_deferred_args(  // NOLINT
      gsl::not_null<Deferred<Rt1>*> deferred, Args&&... args) noexcept;
};

template <typename Rt, typename MakeConstReference>
Deferred<Rt, MakeConstReference>::Deferred(
    boost::shared_ptr<Deferred_detail::assoc_state<tmpl::conditional_t<
        MakeConstReference::value, const value_type&, value_type>>>&&
        state) noexcept
    : state_(std::move(state)) {}

/*!
 * \ingroup DataBoxGroup
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
Deferred<Rt> make_deferred(Fp f, Args&&... args) noexcept {
  return Deferred<Rt>(boost::make_shared<Deferred_detail::deferred_assoc_state<
                          Rt, std::decay_t<Fp>, std::decay_t<Args>...>>(
      f, std::forward<Args>(args)...));
}

namespace Deferred_detail {
template <class Rt>
struct MakeDeferredForSubitemImpl {
  template <typename Fp, typename... Args>
  static Deferred<Rt> apply(Fp f, Args&&... args) noexcept {
    return make_deferred<Rt>(f, std::forward<Args>(args)...);
  }
};

template <class Rt>
struct MakeDeferredForSubitemImpl<const Rt&> {
  template <typename Fp, typename... Args>
  static Deferred<Rt> apply(Fp f, Args&&... args) noexcept {
    return Deferred<Rt>(
        boost::make_shared<Deferred_detail::deferred_assoc_state<
            const Rt&, std::decay_t<Fp>, std::decay_t<Args>...>>(
            f, std::forward<Args>(args)...));
  }
};
}  // namespace Deferred_detail

template <typename Rt, typename Fp, typename... Args>
auto make_deferred_for_subitem(Fp&& f, Args&&... args) noexcept {
  return Deferred_detail::MakeDeferredForSubitemImpl<Rt>::apply(
      std::forward<Fp>(f), std::forward<Args>(args)...);
}

// @{
/*!
 * \ingroup DataBoxGroup
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
                          Fp /*f used for type deduction*/,
                          Args&&... args) noexcept {
  update_deferred_args<Rt, Fp>(deferred, std::forward<Args>(args)...);
}

template <typename Rt, typename Fp, typename... Args>
void update_deferred_args(const gsl::not_null<Deferred<Rt>*> deferred,
                          Args&&... args) noexcept {
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
