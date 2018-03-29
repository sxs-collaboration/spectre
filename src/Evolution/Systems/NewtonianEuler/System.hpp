// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to evolving the Newtonian Euler system
namespace NewtonianEuler {

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
};

}  // namespace NewtonianEuler
