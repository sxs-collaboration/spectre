// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveGr.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"

namespace CurvedScalarWave ::AnalyticData {
template class ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                            gr::Solutions::KerrSchild>;
template bool operator==(const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                                            gr::Solutions::KerrSchild>& lhs,
                         const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                                            gr::Solutions::KerrSchild>& rhs);
template bool operator!=(const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                                            gr::Solutions::KerrSchild>& lhs,
                         const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<3>,
                                            gr::Solutions::KerrSchild>& rhs);
}  // namespace CurvedScalarWave::AnalyticData
