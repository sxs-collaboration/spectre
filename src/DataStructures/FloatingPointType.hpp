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

/// \ingroup DataStructuresGroup
/// \brief Which floating point type to use
///
/// An example use-case is for specifying in an input file to what precision
/// data is written to disk, since most simulations will not have full double
/// precision accuracy on volume data and we don't need all digits to visualize
/// the data.
enum FloatingPointType { Float, Double };

std::ostream& operator<<(std::ostream& os, const FloatingPointType& t) noexcept;

template <>
struct Options::create_from_yaml<FloatingPointType> {
  template <typename Metavariables>
  static FloatingPointType create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
FloatingPointType Options::create_from_yaml<FloatingPointType>::create<void>(
    const Options::Option& options);
