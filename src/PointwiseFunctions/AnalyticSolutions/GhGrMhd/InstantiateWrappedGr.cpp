// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/Prefixes.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/KomissarovShock.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "Utilities/GenerateInstantiations.hpp"

GENERATE_INSTANTIATIONS(WRAPPED_GR_INSTANTIATE,
                        (grmhd::Solutions::AlfvenWave,
                         grmhd::Solutions::BondiMichel,
                         grmhd::Solutions::KomissarovShock,
                         grmhd::Solutions::SmoothFlow))
