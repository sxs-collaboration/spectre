// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
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
      const Variables<prim_tags>& prims);
};

/*!
 * \brief Projects the mass density, velocity, and pressure to the subcells so
 * they can be sent to the neighbors for their reconstructions.
 *
 * Computes the data on the subcells that other generic code will slice before
 * sending it to the neighbors. That is, a `ghost_zone_size * y_points *
 * z_points` is sent to the `x`-direction neighbors.
 *
 * The computation just copies the data from the primitive variables tag to a
 * new Variables, then projects that Variables to the subcells. In the future
 * we will likely want to elide this copy but that requires support from the
 * actions.
 *
 * This mutator is what `Metavars::SubcellOptions::GhostDataToSlice` must be set
 * to.
 *
 * \note This projects the primitive variables rather than computing them on the
 * subcells. This introduces truncation level errors, but from tests so far this
 * seems to be fine and is what is done with local time stepping ADER-DG.
 */
template <size_t Dim>
class PrimitiveGhostDataToSlice {
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
  using argument_tags =
      tmpl::list<::Tags::Variables<prim_tags>, domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>>;

  static Variables<prims_to_reconstruct_tags> apply(
      const Variables<prim_tags>& prims, const Mesh<Dim>& dg_mesh,
      const Mesh<Dim>& subcell_mesh);
};
}  // namespace NewtonianEuler::subcell
