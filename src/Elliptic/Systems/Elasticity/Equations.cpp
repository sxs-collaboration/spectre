// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/Equations.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {

template <size_t Dim>
void primal_fluxes(const gsl::not_null<tnsr::II<DataVector, Dim>*> minus_stress,
                   const tnsr::iJ<DataVector, Dim>& deriv_displacement,
                   const ConstitutiveRelations::ConstitutiveRelation<Dim>&
                       constitutive_relation,
                   const tnsr::I<DataVector, Dim>& coordinates) {
  tnsr::ii<DataVector, Dim> strain{deriv_displacement.begin()->size()};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      strain.get(i, j) =
          0.5 * (deriv_displacement.get(i, j) + deriv_displacement.get(j, i));
    }
  }
  constitutive_relation.stress(minus_stress, strain, coordinates);
  for (auto& component : *minus_stress) {
    component *= -1.;
  }
}

template <size_t Dim>
void add_curved_sources(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> source_for_displacement,
    const tnsr::Ijj<DataVector, Dim>& christoffel_second_kind,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const tnsr::II<DataVector, Dim>& stress) {
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      source_for_displacement->get(j) -=
          christoffel_contracted.get(i) * stress.get(i, j);
      for (size_t k = 0; k < Dim; ++k) {
        source_for_displacement->get(j) -=
            christoffel_second_kind.get(j, i, k) * stress.get(i, k);
      }
    }
  }
}

template <size_t Dim>
void Fluxes<Dim>::apply(
    const gsl::not_null<tnsr::II<DataVector, Dim>*> minus_stress,
    const std::vector<
        std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>>&
        constitutive_relation_per_block,
    const Element<Dim>& element, const tnsr::I<DataVector, Dim>& coordinates,
    const tnsr::I<DataVector, Dim>& /*displacement*/,
    const tnsr::iJ<DataVector, Dim>& deriv_displacement) {
  primal_fluxes(minus_stress, deriv_displacement,
                *constitutive_relation_per_block.at(element.id().block_id()),
                coordinates);
}

template <size_t Dim>
void Fluxes<Dim>::apply(
    const gsl::not_null<tnsr::II<DataVector, Dim>*> minus_stress,
    const std::vector<
        std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>>&
        constitutive_relation_per_block,
    const Element<Dim>& element, const tnsr::I<DataVector, Dim>& coordinates,
    const tnsr::i<DataVector, Dim>& face_normal,
    const tnsr::I<DataVector, Dim>& /*face_normal_vector*/,
    const tnsr::I<DataVector, Dim>& displacement) {
  tnsr::ii<DataVector, Dim> strain{displacement.begin()->size()};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      strain.get(i, j) = 0.5 * (face_normal.get(i) * displacement.get(j) +
                                face_normal.get(j) * displacement.get(i));
    }
  }
  const auto& constitutive_relation =
      *constitutive_relation_per_block.at(element.id().block_id());
  constitutive_relation.stress(minus_stress, strain, coordinates);
  for (auto& component : *minus_stress) {
    component *= -1.;
  }
}

}  // namespace Elasticity

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template void Elasticity::primal_fluxes<DIM(data)>(                    \
      gsl::not_null<tnsr::II<DataVector, DIM(data)>*>,                   \
      const tnsr::iJ<DataVector, DIM(data)>&,                            \
      const Elasticity::ConstitutiveRelations::ConstitutiveRelation<DIM( \
          data)>&,                                                       \
      const tnsr::I<DataVector, DIM(data)>&);                            \
  template void Elasticity::add_curved_sources<DIM(data)>(               \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const tnsr::Ijj<DataVector, DIM(data)>&,                           \
      const tnsr::i<DataVector, DIM(data)>&,                             \
      const tnsr::II<DataVector, DIM(data)>&);                           \
  template class Elasticity::Fluxes<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef INSTANTIATE
#undef DIM
