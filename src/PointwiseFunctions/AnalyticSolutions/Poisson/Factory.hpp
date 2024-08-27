// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/MathFunction.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Zero.hpp"
#include "Utilities/TMPL.hpp"

namespace Poisson::Solutions {
template <size_t Dim, typename DataType = DataVector>
using all_analytic_solutions = tmpl::conditional_t<
    std::is_same_v<DataType, ComplexDataVector>,
    tmpl::flatten<tmpl::list<
        // Only a subset of solutions support ComplexDataVector
        ProductOfSinusoids<Dim, ComplexDataVector>,
        Zero<Dim, ComplexDataVector>,
        tmpl::conditional_t<Dim == 3, Lorentzian<Dim, ComplexDataVector>,
                            tmpl::list<>>>>,
    tmpl::flatten<tmpl::list<
        ProductOfSinusoids<Dim, DataType>, Zero<Dim, DataType>,
        MathFunction<Dim>,
        tmpl::conditional_t<Dim == 1 or Dim == 2, Moustache<Dim>, tmpl::list<>>,
        tmpl::conditional_t<Dim == 3, Lorentzian<Dim, DataType>,
                            tmpl::list<>>>>>;
}  // namespace Poisson::Solutions
