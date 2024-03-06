// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/NewtonianEuler/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/KhInstability.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/ShuOsherTube.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianEuler/SodExplosion.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/SmoothFlow.hpp"

namespace NewtonianEuler::InitialData {
/// The initial data that can be used depending on the spatial and thermodynamic
/// dimension.
template <size_t Dim>
using initial_data_list = tmpl::conditional_t<
    Dim == 3,
    tmpl::list<  // Polytropic EOS
        NewtonianEuler::Solutions::LaneEmdenStar,
        NewtonianEuler::Solutions::IsentropicVortex<3>,

        // Ideal fluid EOS
        NewtonianEuler::Solutions::RiemannProblem<3>,
        NewtonianEuler::Solutions::SmoothFlow<3>,
        AnalyticData::KhInstability<3>, AnalyticData::SodExplosion<3>>,
    tmpl::conditional_t<
        Dim == 2,
        tmpl::list<  // Polytropic EOS
            NewtonianEuler::Solutions::IsentropicVortex<2>,

            // Ideal fluid EOS
            NewtonianEuler::Solutions::RiemannProblem<2>,
            NewtonianEuler::Solutions::SmoothFlow<2>,
            AnalyticData::KhInstability<2>, AnalyticData::SodExplosion<2>>,
        tmpl::list<NewtonianEuler::Solutions::RiemannProblem<1>,
                   NewtonianEuler::Solutions::SmoothFlow<1>,
                   AnalyticData::ShuOsherTube>>>;
}  // namespace NewtonianEuler::InitialData
