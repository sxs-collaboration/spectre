// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.tpp"
#include "Utilities/GenerateInstantiations.hpp"

GENERATE_INSTANTIATIONS(WRAPPED_GR_INSTANTIATE, (gr::Solutions::GaugeWave<1>,
                                                 gr::Solutions::GaugeWave<2>,
                                                 gr::Solutions::GaugeWave<3>,
                                                 gr::Solutions::Minkowski<1>,
                                                 gr::Solutions::Minkowski<2>,
                                                 gr::Solutions::Minkowski<3>,
                                                 gr::Solutions::KerrSchild))
