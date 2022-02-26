// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

#include "Options/Options.hpp"

namespace evolution::dg::subcell::fd {
/*!
 * \brief The reconstruction method to use
 */
enum class ReconstructionMethod {
  /// Dimension-by-dimension reconstruction assuming a tensor-product basis
  DimByDim,
  /// Reconstruct all dimensions at once.
  AllDimsAtOnce
};

std::ostream& operator<<(std::ostream& os, ReconstructionMethod recons_method);
}  // namespace evolution::dg::subcell::fd

/// \cond
template <>
struct Options::create_from_yaml<
    evolution::dg::subcell::fd::ReconstructionMethod> {
  using type = evolution::dg::subcell::fd::ReconstructionMethod;
  template <typename Metavariables>
  static type create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
evolution::dg::subcell::fd::ReconstructionMethod
Options::create_from_yaml<evolution::dg::subcell::fd::ReconstructionMethod>::
    create<void>(const Options::Option& options);
/// \endcond
