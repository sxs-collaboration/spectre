// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"

/// \cond
namespace Poisson {
template <size_t Dim, Geometry BackgroundGeometry>
struct FirstOrderSystem;
}  // namespace Poisson

template <typename System, typename InitialGuess, typename BoundaryConditions>
struct Metavariables;
/// \endcond
