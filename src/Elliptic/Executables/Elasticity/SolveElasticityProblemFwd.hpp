// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace Elasticity {
template <size_t Dim>
struct FirstOrderSystem;
namespace Solutions {
struct BentBeam;
}  // namespace Solutions
}  // namespace Elasticity

template <typename System, typename InitialGuess, typename BoundaryConditions>
struct Metavariables;
/// \endcond
