// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/GrMhd/CcsnCollapse.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/RotatingStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {
/// Analytic solutions of the XCTS equations
namespace Solutions {
using all_analytic_solutions =
    tmpl::list<Flatness, WrappedGr<gr::Solutions::KerrSchild>,
               WrappedGr<gr::Solutions::SphericalKerrSchild>, Schwarzschild,
               WrappedGr<gr::Solutions::HarmonicSchwarzschild>, TovStar,
               WrappedGrMhd<RelativisticEuler::Solutions::RotatingStar>,
               // The following are only approximate solutions to the XCTS
               // equations. We list them here (as opposed to AnalyticData) to
               // avoid a cyclic dependency with the `WrappedGr` class, and also
               // for convenience: by treating them as analytic solutions we can
               // easily measure the difference between the approximate
               // (analytic) solution and the numerical solution.
               WrappedGrMhd<grmhd::AnalyticData::CcsnCollapse>,
               WrappedGrMhd<grmhd::AnalyticData::MagnetizedTovStar>>;
}  // namespace Solutions
}  // namespace Xcts
