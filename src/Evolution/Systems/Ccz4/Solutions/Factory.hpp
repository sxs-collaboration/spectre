// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TrumpetSchwarzschild.hpp"
#include "Utilities/TMPL.hpp"

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
