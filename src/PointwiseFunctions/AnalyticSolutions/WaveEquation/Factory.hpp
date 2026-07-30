// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SecondOrderWrapper.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SemidiscretizedDg.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave::Solutions {
/// \brief List of all analytic solutions
template <size_t Dim>
using all_solutions = tmpl::append<
    tmpl::list<PlaneWave<Dim>>,
    tmpl::conditional_t<Dim == 1, tmpl::list<SemidiscretizedDg>, tmpl::list<>>,
    tmpl::conditional_t<Dim == 3, tmpl::list<RegularSphericalWave>,
                        tmpl::list<>>>;
}  // namespace ScalarWave::Solutions

namespace SecondOrderScalarWave::Solutions {
/// \brief List of all analytic solutions for the second-order in space scalar
/// wave system
template <size_t Dim>
using all_solutions =
    tmpl::list<SecondOrderWrapper<ScalarWave::Solutions::PlaneWave<Dim>>>;
}  // namespace SecondOrderScalarWave::Solutions
