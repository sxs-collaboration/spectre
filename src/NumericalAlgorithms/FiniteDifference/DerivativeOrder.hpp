// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace fd {
/// Controls which FD derivative order is used.
enum class DerivativeOrder : int {
  /// \brief Use one order high derivative.
  ///
  /// For example, if fifth order reconstruction is used, then a sixth-order
  /// derivative is used.
  OneHigherThanRecons = -1,
  /// \brief Same as `OneHigherThanRecons` except uses a fourth-order derivative
  /// if fifth-order reconstruction was used.
  OneHigherThanReconsButFiveToFour = -2,
  /// \brief Use 2nd order derivatives
  Two = 2,
  /// \brief Use 4th order derivatives
  Four = 4,
  /// \brief Use 6th order derivatives
  Six = 6,
  /// \brief Use 8th order derivatives
  Eight = 8,
  /// \brief Use 10th order derivatives
  Ten = 10
};

std::ostream& operator<<(std::ostream& os, DerivativeOrder der_order);
}  // namespace fd

template <>
struct Options::create_from_yaml<fd::DerivativeOrder> {
  template <typename Metavariables>
  static fd::DerivativeOrder create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
fd::DerivativeOrder
Options::create_from_yaml<fd::DerivativeOrder>::create<void>(
    const Options::Option& options);
