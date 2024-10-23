// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Parallel::Callback.

#pragma once

#include <memory>
#include <pup.h>
#include <tuple>
#include <utility>

#include "Parallel/Invoke.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TypeTraits/HasEquivalence.hpp"

namespace Parallel {
namespace detail {
// Not all tuple arguments are guaranteed to have operator==, so we check the
// ones we can.
template <typename... Args>
bool tuple_equal(const std::tuple<Args...>& tuple_1,
                 const std::tuple<Args...>& tuple_2) {
  bool result = true;
  tmpl::for_each<tmpl::make_sequence<tmpl::size_t<0>,
                                     tmpl::size<tmpl::list<Args...>>::value>>(
      [&](const auto index_v) {
        constexpr size_t index = tmpl::type_from<decltype(index_v)>::value;

        if (not result) {
          return;
        }

        if constexpr (tt::has_equivalence_v<decltype(std::get<index>(
                          tuple_1))>) {
          result =
              result and std::get<index>(tuple_1) == std::get<index>(tuple_2);
        }
      });

  return result;
}
}  // namespace detail

/// An abstract base class, whose derived class holds a function that
/// can be invoked at a later time.  The function is intended to be
/// invoked only once.
class Callback : public PUP::able {
 public:
  WRAPPED_PUPable_abstract(Callback);  // NOLINT
  Callback() = default;
  Callback(const Callback&) = default;
  Callback& operator=(const Callback&) = default;
  Callback(Callback&&) = default;
  Callback& operator=(Callback&&) = default;
  ~Callback() override = default;
  explicit Callback(CkMigrateMessage* msg) : PUP::able(msg) {}
  virtual void invoke() = 0;
  virtual void register_with_charm() = 0;
  /*!
   * \brief Returns if this callback is equal to the one passed in.
   */
  virtual bool is_equal_to(const Callback& rhs) const = 0;
  virtual std::string name() const = 0;
  virtual std::unique_ptr<Callback> get_clone() = 0;
};

/// Wraps a call to a simple action and its arguments.
/// Can be invoked only once.
template <typename SimpleAction, typename Proxy, typename... Args>
class SimpleActionCallback : public Callback {
 public:
  WRAPPED_PUPable_decl_template(SimpleActionCallback);  // NOLINT
  SimpleActionCallback() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  SimpleActionCallback(Proxy proxy, std::decay_t<Args>... args)
      : proxy_(proxy), args_(std::move(args)...) {}
  explicit SimpleActionCallback(CkMigrateMessage* msg) : Callback(msg) {}
  using PUP::able::register_constructor;
  void invoke() override {
    std::apply(
        [this](auto&&... args) {
          Parallel::simple_action<SimpleAction>(proxy_, args...);
        },
        std::move(args_));
  }
  void pup(PUP::er& p) override {
    p | proxy_;
    p | args_;
  }

  void register_with_charm() override {
    static bool done_registration{false};
    if (done_registration) {
      return;
    }
    done_registration = true;
    register_classes_with_charm<SimpleActionCallback>();
  }

  bool is_equal_to(const Callback& rhs) const override {
    const auto* downcast_ptr = dynamic_cast<const SimpleActionCallback*>(&rhs);
    if (downcast_ptr == nullptr) {
      return false;
    }
    return detail::tuple_equal(args_, downcast_ptr->args_);
  }

  std::string name() const override {
    // Use pretty_type::get_name with the action since we want to differentiate
    // template paremeters. Only use pretty_type::name for proxy because it'll
    // likely be really long with the template parameters which is unnecessary
    return "SimpleActionCallback(" + pretty_type::get_name<SimpleAction>() +
           "," + pretty_type::name<Proxy>() + ")";
  }

  std::unique_ptr<Callback> get_clone() override {
    return std::make_unique<SimpleActionCallback<SimpleAction, Proxy, Args...>>(
        *this);
  }

 private:
  std::decay_t<Proxy> proxy_{};
  std::tuple<std::decay_t<Args>...> args_{};
};

/// Wraps a call to a simple action without arguments.
template <typename SimpleAction, typename Proxy>
class SimpleActionCallback<SimpleAction, Proxy> : public Callback {
 public:
  WRAPPED_PUPable_decl_template(SimpleActionCallback);  // NOLINT
  SimpleActionCallback() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  SimpleActionCallback(Proxy proxy) : proxy_(proxy) {}
  explicit SimpleActionCallback(CkMigrateMessage* msg) : Callback(msg) {}
  using PUP::able::register_constructor;
  void invoke() override { Parallel::simple_action<SimpleAction>(proxy_); }

  void pup(PUP::er& p) override { p | proxy_; }

  void register_with_charm() override {
    static bool done_registration{false};
    if (done_registration) {
      return;
    }
    done_registration = true;
    register_classes_with_charm<SimpleActionCallback>();
  }

  bool is_equal_to(const Callback& rhs) const override {
    const auto* downcast_ptr = dynamic_cast<const SimpleActionCallback*>(&rhs);
    return downcast_ptr != nullptr;
  }

  std::string name() const override {
    // Use pretty_type::get_name with the action since we want to differentiate
    // template paremeters. Only use pretty_type::name for proxy because it'll
    // likely be really long with the template parameters which is unnecessary
    return "SimpleActionCallback(" + pretty_type::get_name<SimpleAction>() +
           "," + pretty_type::name<Proxy>() + ")";
  }

  std::unique_ptr<Callback> get_clone() override {
    return std::make_unique<SimpleActionCallback<SimpleAction, Proxy>>(*this);
  }

 private:
  std::decay_t<Proxy> proxy_{};
};

/// Wraps a call to a threaded action and its arguments.
/// Can be invoked only once.
template <typename ThreadedAction, typename Proxy, typename... Args>
class ThreadedActionCallback : public Callback {
 public:
  WRAPPED_PUPable_decl_template(ThreadedActionCallback);  // NOLINT
  ThreadedActionCallback() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  ThreadedActionCallback(Proxy proxy, std::decay_t<Args>... args)
      : proxy_(proxy), args_(std::move(args)...) {}
  explicit ThreadedActionCallback(CkMigrateMessage* msg) : Callback(msg) {}
  using PUP::able::register_constructor;
  void invoke() override {
    std::apply(
        [this](auto&&... args) {
          Parallel::threaded_action<ThreadedAction>(proxy_, args...);
        },
        std::move(args_));
  }
  void pup(PUP::er& p) override {
    p | proxy_;
    p | args_;
  }

  void register_with_charm() override {
    static bool done_registration{false};
    if (done_registration) {
      return;
    }
    done_registration = true;
    register_classes_with_charm<ThreadedActionCallback>();
  }

  bool is_equal_to(const Callback& rhs) const override {
    const auto* downcast_ptr =
        dynamic_cast<const ThreadedActionCallback*>(&rhs);
    if (downcast_ptr == nullptr) {
      return false;
    }
    return detail::tuple_equal(args_, downcast_ptr->args_);
  }

  std::string name() const override {
    // Use pretty_type::get_name with the action since we want to differentiate
    // template paremeters. Only use pretty_type::name for proxy because it'll
    // likely be really long with the template parameters which is unnecessary
    return "ThreadedActionCallback(" + pretty_type::get_name<ThreadedAction>() +
           "," + pretty_type::name<Proxy>() + ")";
  }

  std::unique_ptr<Callback> get_clone() override {
    return std::make_unique<
        ThreadedActionCallback<ThreadedAction, Proxy, Args...>>(*this);
  }

 private:
  std::decay_t<Proxy> proxy_{};
  std::tuple<std::decay_t<Args>...> args_{};
};

/// Wraps a call to a threaded action without arguments.
template <typename ThreadedAction, typename Proxy>
class ThreadedActionCallback<ThreadedAction, Proxy> : public Callback {
 public:
  WRAPPED_PUPable_decl_template(ThreadedActionCallback);  // NOLINT
  ThreadedActionCallback() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  ThreadedActionCallback(Proxy proxy) : proxy_(proxy) {}
  explicit ThreadedActionCallback(CkMigrateMessage* msg) : Callback(msg) {}
  using PUP::able::register_constructor;
  void invoke() override { Parallel::threaded_action<ThreadedAction>(proxy_); }

  void pup(PUP::er& p) override { p | proxy_; }

  void register_with_charm() override {
    static bool done_registration{false};
    if (done_registration) {
      return;
    }
    done_registration = true;
    register_classes_with_charm<ThreadedActionCallback>();
  }

  bool is_equal_to(const Callback& rhs) const override {
    const auto* downcast_ptr =
        dynamic_cast<const ThreadedActionCallback*>(&rhs);
    return downcast_ptr != nullptr;
  }

  std::string name() const override {
    // Use pretty_type::get_name with the action since we want to differentiate
    // template paremeters. Only use pretty_type::name for proxy because it'll
    // likely be really long with the template parameters which is unnecessary
    return "ThreadedActionCallback(" + pretty_type::get_name<ThreadedAction>() +
           "," + pretty_type::name<Proxy>() + ")";
  }

  std::unique_ptr<Callback> get_clone() override {
    return std::make_unique<ThreadedActionCallback<ThreadedAction, Proxy>>(
        *this);
  }

 private:
  std::decay_t<Proxy> proxy_{};
};

/// Wraps a call to perform_algorithm.
template <typename Proxy>
class PerformAlgorithmCallback : public Callback {
 public:
  WRAPPED_PUPable_decl_template(PerformAlgorithmCallback);  // NOLINT
  PerformAlgorithmCallback() = default;
  // NOLINTNEXTLINE(google-explicit-constructor)
  PerformAlgorithmCallback(Proxy proxy) : proxy_(proxy) {}
  explicit PerformAlgorithmCallback(CkMigrateMessage* msg) : Callback(msg) {}
  using PUP::able::register_constructor;
  void invoke() override { proxy_.perform_algorithm(); }
  void pup(PUP::er& p) override { p | proxy_; }

  void register_with_charm() override {
    static bool done_registration{false};
    if (done_registration) {
      return;
    }
    done_registration = true;
    register_classes_with_charm<PerformAlgorithmCallback>();
  }

  bool is_equal_to(const Callback& rhs) const override {
    const auto* downcast_ptr =
        dynamic_cast<const PerformAlgorithmCallback*>(&rhs);
    return downcast_ptr != nullptr;
  }

  std::string name() const override {
    // Only use pretty_type::name for proxy because it'll likely be really long
    // with the template parameters which is unnecessary
    return "PerformAlgorithmCallback(" + pretty_type::name<Proxy>() + ")";
  }

  std::unique_ptr<Callback> get_clone() override {
    return std::make_unique<PerformAlgorithmCallback<Proxy>>(*this);
  }

 private:
  std::decay_t<Proxy> proxy_{};
};

/// \cond
template <typename Proxy>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID PerformAlgorithmCallback<Proxy>::my_PUP_ID = 0;
template <typename SimpleAction, typename Proxy, typename... Args>
PUP::able::PUP_ID
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    SimpleActionCallback<SimpleAction, Proxy, Args...>::my_PUP_ID =
        0;  // NOLINT
template <typename SimpleAction, typename Proxy>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID SimpleActionCallback<SimpleAction, Proxy>::my_PUP_ID =
    0;  // NOLINT
template <typename ThreadedAction, typename Proxy, typename... Args>
PUP::able::PUP_ID
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    ThreadedActionCallback<ThreadedAction, Proxy, Args...>::my_PUP_ID =
        0;  // NOLINT
template <typename ThreadedAction, typename Proxy>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID ThreadedActionCallback<ThreadedAction, Proxy>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Parallel
