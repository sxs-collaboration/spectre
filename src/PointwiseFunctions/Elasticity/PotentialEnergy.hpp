// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

namespace Elasticity {

template <size_t Dim>
Scalar<DataVector> evaluate_potential_energy(
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& strain,
    const tnsr::I<DataVector, Dim>& coordinates,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) noexcept;

namespace Tags {
template <size_t Dim>
struct PotentialEnergyCompute : Elasticity::Tags::PotentialEnergy<Dim>,
                                db::ComputeTag {
  using base = Elasticity::Tags::PotentialEnergy<Dim>;
  using argument_tags =
      tmpl::list<Elasticity::Tags::Strain<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 Elasticity::Tags::ConstitutiveRelationBase>;
  // static constexpr auto function = &evaluate_potential_energy<Dim>;
  static Scalar<DataVector> function(
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& strain,
      const tnsr::I<DataVector, Dim>& coordinates,
      const Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>&
          constitutive_relation) {
    return evaluate_potential_energy<Dim>(strain, coordinates,
                                          constitutive_relation);
  }
};
}  // namespace Tags
}  // namespace Elasticity
