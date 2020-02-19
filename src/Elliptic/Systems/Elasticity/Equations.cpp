// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {

template <size_t Dim>
void primal_fluxes(
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> flux_for_displacement,
    const tnsr::ii<DataVector, Dim>& strain,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation,
    const tnsr::I<DataVector, Dim>& coordinates) noexcept {
  const auto stress = constitutive_relation.stress(strain, coordinates);
  // To set the components of the flux each component of the symmetric stress
  // tensor is used twice. So the tensor can't be moved in its entirety.
  for (size_t d = 0; d < Dim; d++) {
    for (size_t e = 0; e < Dim; e++) {
      // Also, the stress has lower and the flux upper indices. The minus sign
      // originates in the definition of the stress \f$T^{ij} = -Y^{ijkl}
      // S_{kl}\f$.
      flux_for_displacement->get(d, e) = -stress.get(d, e);
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

}  // namespace Elasticity

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template void Elasticity::primal_fluxes<DIM(data)>(                    \
      gsl::not_null<tnsr::IJ<DataVector, DIM(data)>*>,                   \
      const tnsr::ii<DataVector, DIM(data)>&,                            \
      const Elasticity::ConstitutiveRelations::ConstitutiveRelation<DIM( \
          data)>&,                                                       \
      const tnsr::I<DataVector, DIM(data)>&) noexcept;                   \
  template void Elasticity::auxiliary_fluxes<DIM(data)>(                 \
      gsl::not_null<tnsr::Ijj<DataVector, DIM(data)>*>,                  \
      const tnsr::I<DataVector, DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

// Instantiate derivative templates
template <size_t Dim>
using variables_tag = typename Elasticity::FirstOrderSystem<Dim>::variables_tag;
template <size_t Dim>
using fluxes_tags_list = db::get_variables_tags_list<db::add_tag_prefix<
    ::Tags::Flux, variables_tag<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;

#define INSTANTIATE_DERIVS(_, data)                                            \
  template Variables<db::wrap_tags_in<Tags::div, fluxes_tags_list<DIM(data)>>> \
  divergence<fluxes_tags_list<DIM(data)>, DIM(data), Frame::Inertial>(         \
      const Variables<fluxes_tags_list<DIM(data)>>&, const Mesh<DIM(data)>&,   \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,             \
                            Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_DERIVS, (2, 3))

#undef INSTANTIATE
#undef INSTANTIATE_DERIVS
#undef DIM
