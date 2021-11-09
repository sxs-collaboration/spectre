// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Krivodonova.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Kuzmin.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Sinusoid.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::Solutions {
/*!
 * \brief Typelist of all analytic solutions of advection system
 */
template <size_t Dim>
using all_analytic_solutions = tmpl::flatten<tmpl::list<
    tmpl::conditional_t<Dim == 1,
                        tmpl::list<ScalarAdvection::Solutions::Sinusoid,
                                   ScalarAdvection::Solutions::Krivodonova>,
                        tmpl::list<>>,
    tmpl::conditional_t<Dim == 2,
                        tmpl::list<ScalarAdvection::Solutions::Kuzmin>,
                        tmpl::list<>>>>;
}  // namespace ScalarAdvection::Solutions
