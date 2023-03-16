// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace gr::Solutions {
class KerrSchild;
}  // namespace gr::Solutions

namespace ScalarWave::Solutions {
template <size_t Dim>
class PlaneWave;
}  // namespace ScalarWave::Solutions

template <typename BackgroundSpacetime, typename InitialData>
struct EvolutionMetavars;
/// \endcond
