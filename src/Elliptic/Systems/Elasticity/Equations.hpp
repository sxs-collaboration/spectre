// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace Elasticity {
namespace ConstitutiveRelations {
template <size_t Dim>
class ConstitutiveRelation;
}  // namespace ConstitutiveRelations
}  // namespace Elasticity
/// \endcond

namespace Elasticity {

/*!
 * \brief Compute the fluxes \f$F^{ij}=Y^{ijkl}(x) S_{kl}(x)=-T^{ij}\f$ for
 * the Elasticity equation.
 */
template <size_t Dim>
void primal_fluxes(gsl::not_null<tnsr::II<DataVector, Dim>*> minus_stress,
                   const tnsr::iJ<DataVector, Dim>& deriv_displacement,
                   const ConstitutiveRelations::ConstitutiveRelation<Dim>&
                       constitutive_relation,
                   const tnsr::I<DataVector, Dim>& coordinates);

/*!
 * \brief Add the contribution \f$-\Gamma^i_{ik}T^{kj} - \Gamma^j_{ik}T^{ik}\f$
 * to the displacement source for the curved-space elasticity equations on a
 * metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the divergence on a
 * curved background.
 */
template <size_t Dim>
void add_curved_sources(
    gsl::not_null<tnsr::I<DataVector, Dim>*> source_for_displacement,
    const tnsr::Ijj<DataVector, Dim>& christoffel_second_kind,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const tnsr::II<DataVector, Dim>& stress);

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the Elasticity equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Elasticity::FirstOrderSystem
 */
template <size_t Dim>
struct Fluxes {
  using argument_tags =
      tmpl::list<Tags::ConstitutiveRelationPerBlockBase,
                 domain::Tags::Element<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using volume_tags = tmpl::list<Tags::ConstitutiveRelationPerBlockBase,
                                 domain::Tags::Element<Dim>>;
  using const_global_cache_tags =
      tmpl::list<Tags::ConstitutiveRelationPerBlockBase>;
  static void apply(
      gsl::not_null<tnsr::II<DataVector, Dim>*> minus_stress,
      const std::vector<
          std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>>&
          constitutive_relation_per_block,
      const Element<Dim>& element, const tnsr::I<DataVector, Dim>& coordinates,
      const tnsr::I<DataVector, Dim>& displacement,
      const tnsr::iJ<DataVector, Dim>& deriv_displacement);
  static void apply(
      gsl::not_null<tnsr::II<DataVector, Dim>*> minus_stress,
      const std::vector<
          std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>>&
          constitutive_relation_per_block,
      const Element<Dim>& element, const tnsr::I<DataVector, Dim>& coordinates,
      const tnsr::i<DataVector, Dim>& face_normal,
      const tnsr::I<DataVector, Dim>& face_normal_vector,
      const tnsr::I<DataVector, Dim>& displacement);
};

}  // namespace Elasticity
