// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {
/// Analytic solutions of the XCTS equations
namespace Solutions {
using all_analytic_solutions =
    tmpl::list<Flatness, WrappedGr<gr::Solutions::KerrSchild>,
               WrappedGr<gr::Solutions::SphericalKerrSchild>, Schwarzschild,
               WrappedGr<gr::Solutions::HarmonicSchwarzschild>>;
}  // namespace Solutions
}  // namespace Xcts
