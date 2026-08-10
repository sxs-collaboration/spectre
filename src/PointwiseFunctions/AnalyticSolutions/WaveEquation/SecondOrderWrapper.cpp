// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SecondOrderWrapper.hpp"

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SecondOrderWrapper.tpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/StandingWave.hpp"
#include "Utilities/GenerateInstantiations.hpp"

GENERATE_INSTANTIATIONS(SECOND_ORDER_WRAPPER_INSTANTIATE,
                        (ScalarWave::Solutions::PlaneWave<1>,
                         ScalarWave::Solutions::PlaneWave<2>,
                         ScalarWave::Solutions::PlaneWave<3>,
                         ScalarWave::Solutions::StandingWave<1>,
                         ScalarWave::Solutions::StandingWave<2>,
                         ScalarWave::Solutions::StandingWave<3>))
