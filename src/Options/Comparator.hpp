// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/ErrorHandling/Error.hpp"

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Options {
/// An option-creatable mathematical comparison.
class Comparator {
 public:
  enum class Comparison {
    EqualTo,
    NotEqualTo,
    LessThan,
    GreaterThan,
    LessThanOrEqualTo,
    GreaterThanOrEqualTo
  };

  Comparator() noexcept = default;
  Comparator(Comparison comparison) noexcept;

  template <typename T1, typename T2>
  bool operator()(const T1& t1, const T2& t2) const noexcept {
    switch (comparison_) {
      case Comparison::EqualTo:
        return t1 == t2;
      case Comparison::NotEqualTo:
        return t1 != t2;
      case Comparison::LessThan:
        return t1 < t2;
      case Comparison::GreaterThan:
        return t1 > t2;
      case Comparison::LessThanOrEqualTo:
        return t1 <= t2;
      case Comparison::GreaterThanOrEqualTo:
        return t1 >= t2;
      default:
        ERROR("Invalid comparison");
    }
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  Comparison comparison_;
};

template <>
struct create_from_yaml<Comparator> {
  template <typename Metavariables>
  static Comparator create(const Option& options) {
    return create_impl(options);
  }

 private:
  static Comparator create_impl(const Option& options);
};
}  // namespace Options
