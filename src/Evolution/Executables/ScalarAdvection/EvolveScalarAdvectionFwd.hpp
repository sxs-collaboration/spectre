// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace ScalarAdvection {
namespace Solutions {
class Krivodonova;
class Kuzmin;
class Sinusoid;
}  // namespace Solutions
}  // namespace ScalarAdvection

template <size_t Dim, typename InitialData>
struct EvolutionMetavars;
