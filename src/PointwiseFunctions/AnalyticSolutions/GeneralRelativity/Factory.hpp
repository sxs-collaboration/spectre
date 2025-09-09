// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Ccz4WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugePlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TrumpetSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::Solutions {
/// \brief List of all GH analytic solutions
template <size_t Dim>
using all_solutions =
    tmpl::append<tmpl::list<WrappedGr<gr::Solutions::GaugePlaneWave<Dim>>,
                            WrappedGr<gr::Solutions::GaugeWave<Dim>>,
                            WrappedGr<gr::Solutions::Minkowski<Dim>>>,
                 tmpl::conditional_t<
                     Dim == 3,
                     tmpl::list<WrappedGr<gr::Solutions::HarmonicSchwarzschild>,
                                WrappedGr<gr::Solutions::KerrSchild>,
                                WrappedGr<gr::Solutions::SphericalKerrSchild>,
                                WrappedGr<gr::Solutions::TrumpetSchwarzschild>>,
                     tmpl::list<>>>;
}  // namespace gh::Solutions

namespace Ccz4::Solutions {
/// \brief List of all Ccz4 analytic solutions
/// Right now it only makes sense to do TrumpetSchwarzschild or time-independent
/// solutions because we can either use 1+log slicing with Gamma-driver
/// or not evolve the lapse and shift at all. We will allow analytic
/// lapse and shift in the future.
using all_solutions =
    tmpl::list<Ccz4WrappedGr<gr::Solutions::KerrSchild>,
               Ccz4WrappedGr<gr::Solutions::Minkowski<3>>,
               Ccz4WrappedGr<gr::Solutions::TrumpetSchwarzschild>>;
}  // namespace Ccz4::Solutions
