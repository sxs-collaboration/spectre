// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/Prefixes.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BondiHoyleAccretion.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.tpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"

GENERATE_INSTANTIATIONS(WRAPPED_GR_INSTANTIATE,
                        (grmhd::AnalyticData::BondiHoyleAccretion,
                         grmhd::AnalyticData::MagnetizedFmDisk,
                         grmhd::AnalyticData::MagnetizedTovStar))
