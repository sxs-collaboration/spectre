// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ScalarWaveSystem.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Evolution/Systems/ScalarWave/Equations.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
template <class>
class Variables;
}  // namespace Tags

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the scalar wave equation:
 */
namespace ScalarWave {

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;

  using variables_tag = Tags::Variables<tmpl::list<Pi, Phi<Dim>, Psi>>;
  // Typelist of which subset of the variables to take the gradient of.
  using gradients_tags = tmpl::list<Pi, Phi<Dim>>;

  using du_dt = ComputeDuDt<Dim>;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed;

  // This is only used in InitializeElement
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};
}  // namespace ScalarWave
