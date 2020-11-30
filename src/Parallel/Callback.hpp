// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Parallel::Callback.

#pragma once

#include <pup.h>
#include <tuple>
#include <utility>

#include "Parallel/CharmPupable.hpp"
#include "Parallel/Invoke.hpp"

namespace Parallel {
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
  virtual ~Callback() = default;
  virtual void invoke() noexcept = 0;
};

/// Wraps a call to a simple action and its arguments.
/// Can be invoked only once.
template <typename SimpleAction, typename Proxy, typename... Args>
class SimpleActionCallback : public Callback {
 public:
  WRAPPED_PUPable_decl_template(SimpleActionCallback);  // NOLINT
  SimpleActionCallback(Proxy proxy, Args&&... args)
      : proxy_(proxy), args_(std::make_tuple<Args...>(args...)) {}
  void invoke() noexcept override {
    std::apply(
        [this](auto&&... args) noexcept {
          Parallel::simple_action<SimpleAction>(proxy_, args...);
        },
        std::move(args_));
  }
  void pup(PUP::er& p) noexcept override {
    p | proxy_;
    p | args_;
  }

 private:
  Proxy proxy_;
  std::tuple<Args...> args_;
};

/// Wraps a call to a simple action without arguments.
template <typename SimpleAction, typename Proxy>
class SimpleActionCallback<SimpleAction, Proxy> : public Callback {
 public:
  WRAPPED_PUPable_decl_template(SimpleActionCallback);  // NOLINT
  SimpleActionCallback(Proxy proxy) : proxy_(proxy) {}
  void invoke() noexcept override {
    Parallel::simple_action<SimpleAction>(proxy_);
  }

  void pup(PUP::er& p) noexcept override { p | proxy_; }

 private:
  Proxy proxy_;
};

/// Wraps a call to perform_algorithm.
template <typename Proxy>
class PerformAlgorithmCallback : public Callback {
 public:
  WRAPPED_PUPable_decl_template(PerformAlgorithmCallback);  // NOLINT
  PerformAlgorithmCallback(Proxy proxy) : proxy_(proxy) {}
  void invoke() noexcept override { proxy_.perform_algorithm(); }
  void pup(PUP::er& p) noexcept override { p | proxy_; }

 private:
  Proxy proxy_;
};

/// \cond
template <typename Proxy>
PUP::able::PUP_ID PerformAlgorithmCallback<Proxy>::my_PUP_ID = 0;  // NOLINT
template <typename SimpleAction, typename Proxy, typename... Args>
PUP::able::PUP_ID
    SimpleActionCallback<SimpleAction, Proxy, Args...>::my_PUP_ID =
        0;  // NOLINT
template <typename SimpleAction, typename Proxy>
PUP::able::PUP_ID SimpleActionCallback<SimpleAction, Proxy>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Parallel
