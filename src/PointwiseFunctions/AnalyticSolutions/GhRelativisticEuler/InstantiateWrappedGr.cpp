// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/Prefixes.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.tpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"

GENERATE_INSTANTIATIONS(
    WRAPPED_GR_INSTANTIATE,
    (RelativisticEuler::Solutions::FishboneMoncriefDisk,
     RelativisticEuler::Solutions::SmoothFlow<1>,
     RelativisticEuler::Solutions::SmoothFlow<2>,
     RelativisticEuler::Solutions::SmoothFlow<3>,
     RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution>))
