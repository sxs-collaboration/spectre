// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/Equations.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {

template <size_t Dim>
void primal_fluxes(
    const gsl::not_null<tnsr::II<DataVector, Dim>*> flux_for_displacement,
    const tnsr::ii<DataVector, Dim>& strain,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation,
    const tnsr::I<DataVector, Dim>& coordinates) noexcept {
  constitutive_relation.stress(flux_for_displacement, strain, coordinates);
  for (auto& component : *flux_for_displacement) {
    component *= -1.;
  }
}

template <size_t Dim>
void add_curved_sources(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> source_for_displacement,
    const tnsr::Ijj<DataVector, Dim>& christoffel_second_kind,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const tnsr::II<DataVector, Dim>& stress) noexcept {
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
void auxiliary_fluxes(
    const gsl::not_null<tnsr::Ijj<DataVector, Dim>*> flux_for_strain,
    const tnsr::I<DataVector, Dim>& displacement) noexcept {
  std::fill(flux_for_strain->begin(), flux_for_strain->end(), 0.);
  // The off-diagonal elements are calculated by going over the upper triangular
  // matrix (the lower triangular matrix, excluding the diagonal elements, is
  // set by virtue of the tensor being symmetric in its last two indices) and
  // the symmetrisation is completed by going over the diagonal elements again.
  for (size_t d = 0; d < Dim; d++) {
    flux_for_strain->get(d, d, d) += 0.5 * displacement.get(d);
    for (size_t e = 0; e < Dim; e++) {
      flux_for_strain->get(d, e, d) += 0.5 * displacement.get(e);
    }
  }
}

template <size_t Dim>
void curved_auxiliary_fluxes(
    const gsl::not_null<tnsr::Ijj<DataVector, Dim>*> flux_for_strain,
    const tnsr::ii<DataVector, Dim>& metric,
    const tnsr::I<DataVector, Dim>& displacement) noexcept {
  const auto co_displacement = raise_or_lower_index(displacement, metric);
  std::fill(flux_for_strain->begin(), flux_for_strain->end(), 0.);
  for (size_t d = 0; d < Dim; ++d) {
    flux_for_strain->get(d, d, d) += 0.5 * co_displacement.get(d);
    for (size_t e = 0; e < Dim; ++e) {
      flux_for_strain->get(d, e, d) += 0.5 * co_displacement.get(e);
    }
  }
}

template <size_t Dim>
void add_curved_auxiliary_sources(
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> source_for_strain,
    const tnsr::ijj<DataVector, Dim>& christoffel_first_kind,
    const tnsr::I<DataVector, Dim>& displacement) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      for (size_t k = 0; k < Dim; ++k) {
        source_for_strain->get(i, j) +=
            christoffel_first_kind.get(k, i, j) * displacement.get(k);
      }
    }
  }
}

/// \cond
template <size_t Dim>
void Fluxes<Dim>::apply(
    const gsl::not_null<tnsr::II<DataVector, Dim>*> flux_for_displacement,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation,
    const tnsr::I<DataVector, Dim>& coordinates,
    const tnsr::ii<DataVector, Dim>& strain) noexcept {
  primal_fluxes(flux_for_displacement, strain, constitutive_relation,
                coordinates);
}

template <size_t Dim>
void Fluxes<Dim>::apply(
    const gsl::not_null<tnsr::Ijj<DataVector, Dim>*> flux_for_strain,
    const ConstitutiveRelations::ConstitutiveRelation<
        Dim>& /*constitutive_relation*/,
    const tnsr::I<DataVector, Dim>& /*coordinates*/,
    const tnsr::I<DataVector, Dim>& displacement) noexcept {
  auxiliary_fluxes(flux_for_strain, displacement);
}

template <size_t Dim>
void Sources<Dim>::apply(
    const gsl::not_null<
        tnsr::I<DataVector, Dim>*> /*equation_for_displacement*/,
    const tnsr::I<DataVector, Dim>& /*displacement*/,
    const tnsr::II<DataVector, Dim>& /*minus_stress*/) noexcept {}

template <size_t Dim>
void Sources<Dim>::apply(
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> /*equation_for_strain*/,
    const tnsr::I<DataVector, Dim>& /*displacement*/) noexcept {}
/// \endcond

}  // namespace Elasticity

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template void Elasticity::primal_fluxes<DIM(data)>(                    \
      gsl::not_null<tnsr::II<DataVector, DIM(data)>*>,                   \
      const tnsr::ii<DataVector, DIM(data)>&,                            \
      const Elasticity::ConstitutiveRelations::ConstitutiveRelation<DIM( \
          data)>&,                                                       \
      const tnsr::I<DataVector, DIM(data)>&) noexcept;                   \
  template void Elasticity::add_curved_sources<DIM(data)>(               \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const tnsr::Ijj<DataVector, DIM(data)>&,                           \
      const tnsr::i<DataVector, DIM(data)>&,                             \
      const tnsr::II<DataVector, DIM(data)>&) noexcept;                  \
  template void Elasticity::auxiliary_fluxes<DIM(data)>(                 \
      gsl::not_null<tnsr::Ijj<DataVector, DIM(data)>*>,                  \
      const tnsr::I<DataVector, DIM(data)>&) noexcept;                   \
  template void Elasticity::curved_auxiliary_fluxes<DIM(data)>(          \
      gsl::not_null<tnsr::Ijj<DataVector, DIM(data)>*>,                  \
      const tnsr::ii<DataVector, DIM(data)>&,                            \
      const tnsr::I<DataVector, DIM(data)>&) noexcept;                   \
  template void Elasticity::add_curved_auxiliary_sources<DIM(data)>(     \
      gsl::not_null<tnsr::ii<DataVector, DIM(data)>*>,                   \
      const tnsr::ijj<DataVector, DIM(data)>&,                           \
      const tnsr::I<DataVector, DIM(data)>&) noexcept;                   \
  template class Elasticity::Sources<DIM(data)>;                         \
  template class Elasticity::Fluxes<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef INSTANTIATE
#undef DIM
