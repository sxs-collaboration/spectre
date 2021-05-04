// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename TagsList>
class Variables;
/// \endcond

namespace NewtonianEuler::subcell {
/*!
 * \brief Computes the mass density, velocity, and pressure on the subcells so
 * they can be sent to the neighbors for their reconstructions.
 *
 * The computation just copies the data from the primitive variables tag to a
 * new Variables (the copy is subcell grid to subcell grid). In the future we
 * will likely want to elide this copy but that requires support from the
 * actions.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 */
template <size_t Dim>
class PrimitiveGhostDataOnSubcells {
 private:
  using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
  using Velocity = NewtonianEuler::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;

  using prim_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using prims_to_reconstruct_tags = tmpl::list<MassDensity, Velocity, Pressure>;

 public:
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::Variables<prim_tags>>;

  static Variables<prims_to_reconstruct_tags> apply(
      const Variables<prim_tags>& prims) noexcept;
};
}  // namespace NewtonianEuler::subcell
