// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace elliptic {

/// Identify types of boundary conditions for elliptic equations
enum class BoundaryConditionType {
  /// Dirichlet boundary conditions like \f$u(x_0)=u_0\f$
  Dirichlet,
  /// Neumann boundary conditions like \f$n^i\partial_i u(x_0)=v_0\f$, where
  /// \f$\boldsymbol{n}\f$ is the normal to the domain boundary
  Neumann
};

std::ostream& operator<<(
    std::ostream& os, BoundaryConditionType boundary_condition_type) noexcept;

}  // namespace elliptic

/// \cond
template <>
struct Options::create_from_yaml<elliptic::BoundaryConditionType> {
  template <typename Metavariables>
  static elliptic::BoundaryConditionType create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
elliptic::BoundaryConditionType
Options::create_from_yaml<elliptic::BoundaryConditionType>::create<void>(
    const Options::Option& options);
/// \endcond
