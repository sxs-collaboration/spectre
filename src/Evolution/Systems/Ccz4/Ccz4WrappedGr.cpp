// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"

#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugePlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TrumpetSchwarzschild.hpp"
#include "Utilities/GenerateInstantiations.hpp"

GENERATE_INSTANTIATIONS(CCZ4_WRAPPED_GR_INSTANTIATE,
                        (gr::Solutions::GaugeWave<3>,
                         gr::Solutions::GaugePlaneWave<3>,
                         gr::Solutions::KerrSchild, gr::Solutions::Minkowski<3>,
                         gr::Solutions::TrumpetSchwarzschild))
