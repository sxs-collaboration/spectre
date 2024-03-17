// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
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
class PrimitiveGhostVariables {
 private:
  using MassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using Velocity = hydro::Tags::SpatialVelocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      hydro::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = hydro::Tags::Pressure<DataVector>;

  using prim_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using prims_to_reconstruct_tags = tmpl::list<MassDensity, Velocity, Pressure>;

 public:
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::Variables<prim_tags>>;

  static DataVector apply(const Variables<prim_tags>& prims, size_t rdmp_size);
};
}  // namespace NewtonianEuler::subcell
