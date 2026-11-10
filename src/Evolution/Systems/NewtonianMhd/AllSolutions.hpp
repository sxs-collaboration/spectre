// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/NewtonianMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianMhd/KhInstability.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianMhd/ShuOsherTube.hpp"
#include "PointwiseFunctions/AnalyticData/NewtonianMhd/SodExplosion.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/IsentropicVortex.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/LaneEmdenStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianMhd/SmoothFlow.hpp"

namespace NewtonianMhd::InitialData {
/// The initial data that can be used depending on the spatial and thermodynamic
/// dimension.
template <size_t Dim>
using initial_data_list = tmpl::conditional_t<
    Dim == 3,
    tmpl::list<  // Polytropic EOS
        NewtonianMhd::Solutions::LaneEmdenStar,
        NewtonianMhd::Solutions::IsentropicVortex<3>,

        // Ideal fluid EOS
        NewtonianMhd::Solutions::RiemannProblem<3>,
        NewtonianMhd::Solutions::SmoothFlow<3>, AnalyticData::KhInstability<3>,
        AnalyticData::SodExplosion<3>>,
    tmpl::conditional_t<
        Dim == 2,
        tmpl::list<  // Polytropic EOS
            NewtonianMhd::Solutions::IsentropicVortex<2>,

            // Ideal fluid EOS
            NewtonianMhd::Solutions::RiemannProblem<2>,
            NewtonianMhd::Solutions::SmoothFlow<2>,
            AnalyticData::KhInstability<2>, AnalyticData::SodExplosion<2>>,
        tmpl::list<NewtonianMhd::Solutions::RiemannProblem<1>,
                   NewtonianMhd::Solutions::SmoothFlow<1>,
                   AnalyticData::ShuOsherTube>>>;
}  // namespace NewtonianMhd::InitialData
