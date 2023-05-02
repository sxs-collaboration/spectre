// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::Solutions {
/// \brief List of all analytic solutions
template <size_t Dim>
using all_solutions =
    tmpl::append<tmpl::list<WrappedGr<gr::Solutions::GaugeWave<Dim>>,
                            WrappedGr<gr::Solutions::Minkowski<Dim>>>,
                 tmpl::conditional_t<
                     Dim == 3,
                     tmpl::list<WrappedGr<gr::Solutions::HarmonicSchwarzschild>,
                                WrappedGr<gr::Solutions::KerrSchild>,
                                WrappedGr<gr::Solutions::SphericalKerrSchild>>,
                     tmpl::list<>>>;
}  // namespace gh::Solutions
