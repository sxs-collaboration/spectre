// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

namespace Elasticity::Tags {

template <size_t Dim>
struct StressCompute : Elasticity::Tags::Stress<Dim>, db::ComputeTag {
  using base = Elasticity::Tags::Stress<Dim>;
  using return_type = tnsr::II<DataVector, Dim>;
  using argument_tags =
      tmpl::list<Elasticity::Tags::Strain<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 Elasticity::Tags::ConstitutiveRelation<Dim>>;
  static void function(const gsl::not_null<tnsr::II<DataVector, Dim>*> stress,
                       const tnsr::ii<DataVector, Dim>& strain,
                       const tnsr::I<DataVector, Dim>& coordinates,
                       const ConstitutiveRelations::ConstitutiveRelation<Dim>&
                           constitutive_relation) {
    constitutive_relation.stress(stress, strain, coordinates);
  }
};

}  // namespace Elasticity::Tags
