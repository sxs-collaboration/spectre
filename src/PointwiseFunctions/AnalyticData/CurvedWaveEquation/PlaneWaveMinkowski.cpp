// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/ScalarWaveGr.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"

namespace CurvedScalarWave::AnalyticData {
#define FUNCS_DECL(dim)                                              \
  template class ScalarWaveGr<ScalarWave::Solutions::PlaneWave<dim>, \
                              gr::Solutions::Minkowski<dim>>;        \
  template bool operator==(                                          \
      const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<dim>,      \
                         gr::Solutions::Minkowski<dim>>& lhs,        \
      const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<dim>,      \
                         gr::Solutions::Minkowski<dim>>& rhs);       \
  template bool operator!=(                                          \
      const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<dim>,      \
                         gr::Solutions::Minkowski<dim>>& lhs,        \
      const ScalarWaveGr<ScalarWave::Solutions::PlaneWave<dim>,      \
                         gr::Solutions::Minkowski<dim>>& rhs);

FUNCS_DECL(1)
FUNCS_DECL(2)
FUNCS_DECL(3)
#undef FUNCS_DECL
}  // namespace CurvedScalarWave::AnalyticData
