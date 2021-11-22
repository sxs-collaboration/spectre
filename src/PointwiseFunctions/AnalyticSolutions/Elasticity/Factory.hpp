// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/Zero.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::Solutions {
template <size_t Dim>
using all_analytic_solutions = tmpl::append<
    tmpl::list<Zero<Dim>>,
    tmpl::conditional_t<Dim == 2, tmpl::list<BentBeam>, tmpl::list<>>,
    tmpl::conditional_t<Dim == 3, tmpl::list<HalfSpaceMirror>, tmpl::list<>>>;
}  // namespace Elasticity::Solutions
