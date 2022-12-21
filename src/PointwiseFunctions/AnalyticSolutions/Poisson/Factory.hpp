// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/MathFunction.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Zero.hpp"
#include "Utilities/TMPL.hpp"

namespace Poisson::Solutions {
template <size_t Dim>
using all_analytic_solutions = tmpl::append<
    tmpl::list<ProductOfSinusoids<Dim>, Zero<Dim>, MathFunction<Dim>>,
    tmpl::conditional_t<Dim == 1 or Dim == 2, tmpl::list<Moustache<Dim>>,
                        tmpl::list<>>,
    tmpl::conditional_t<Dim == 3, tmpl::list<Lorentzian<Dim>>, tmpl::list<>>>;
}  // namespace Poisson::Solutions
