// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>

#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief Holds resource info for a single singleton component
 *
 * \details Holds what proc the singleton is to be placed on and whether that
 * proc should be exclusive, i.e. no array component elements or other
 * singletons placed on that proc. Instead of specifying a proc, the proc can be
 * chosen automatically by using the `Options::Auto` option.
 */
struct SingletonInfoHolder {
  struct Proc {
    using type = Options::Auto<int>;
    static constexpr Options::String help = {
        "Proc to put singleton on. This can be determined automatically if "
        "desired by specifying 'Auto' (without quotes)."};
  };

  struct Exclusive {
    using type = bool;
    static constexpr Options::String help = {
        "Reserve this proc for this singleton. No array component elements or "
        "other singleton components will be placed on this proc."};
  };

  using options = tmpl::list<Proc, Exclusive>;
  static constexpr Options::String help = {
      "Resource options for a single singleton."};

  SingletonInfoHolder(std::optional<int> input_proc, const bool input_exclusive,
                      const Options::Context& context = {})
      : exclusive_(input_exclusive) {
    // If there is no value, we don't need to error so use 0 as a comparator
    // in both cases
    if (input_proc.value_or(0) < 0) {
      PARSE_ERROR(
          context,
          "Proc must be a non-negative integer. Please choose another proc.");
    }

    proc_ = input_proc.has_value()
                ? std::optional<size_t>(static_cast<size_t>(input_proc.value()))
                : std::nullopt;
  }

  SingletonInfoHolder() = default;
  SingletonInfoHolder(const SingletonInfoHolder& /*rhs*/) = default;
  SingletonInfoHolder& operator=(const SingletonInfoHolder& /*rhs*/) = default;
  SingletonInfoHolder(SingletonInfoHolder&& /*rhs*/) = default;
  SingletonInfoHolder& operator=(SingletonInfoHolder&& /*rhs*/) = default;
  ~SingletonInfoHolder() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | proc_;
    p | exclusive_;
  };

  /// Proc that singleton is to be placed on. If the optional is a std::nullopt,
  /// then the proc should be chosen automatically.
  std::optional<size_t> proc() const { return proc_; }

  /// Whether or not the singleton wants to be exclusive on the proc.
  bool is_exclusive() const { return exclusive_; }

 private:
  // We use size_t here because we want a non-negative integer, but we use int
  // in the option because we want to protect against negative numbers. And a
  // negative size_t is actually a really large value (it wraps around)
  std::optional<size_t> proc_{std::nullopt};
  bool exclusive_{false};
};
}  // namespace Parallel
