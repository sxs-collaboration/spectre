// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BlastWave.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/BondiHoyleAccretion.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/CcsnCollapse.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/KhInstability.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticFieldLoop.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagneticRotor.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/OrszagTangVortex.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/PolarMagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/RiemannProblem.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/SlabJet.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/KomissarovShock.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/Solutions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"

namespace grmhd::ValenciaDivClean::InitialData {
using initial_data_list = tmpl::list<
    AnalyticData::BlastWave, AnalyticData::BondiHoyleAccretion,
    AnalyticData::CcsnCollapse, AnalyticData::KhInstability,
    AnalyticData::MagneticFieldLoop, AnalyticData::MagneticRotor,
    AnalyticData::MagnetizedFmDisk, AnalyticData::MagnetizedTovStar,
    AnalyticData::OrszagTangVortex, AnalyticData::PolarMagnetizedFmDisk,
    AnalyticData::RiemannProblem, AnalyticData::SlabJet, Solutions::AlfvenWave,
    grmhd::Solutions::BondiMichel, Solutions::KomissarovShock,
    Solutions::SmoothFlow, RelativisticEuler::Solutions::FishboneMoncriefDisk,
    RelativisticEuler::Solutions::RotatingStar,
    RelativisticEuler::Solutions::TovStar>;
}  // namespace grmhd::ValenciaDivClean::InitialData
