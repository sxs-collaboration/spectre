// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"

/// \cond
namespace Elasticity {
template <size_t Dim>
struct FirstOrderSystem;
}  // namespace Elasticity

template <typename System, typename InitialGuess, typename BoundaryConditions>
struct Metavariables;
/// \endcond
