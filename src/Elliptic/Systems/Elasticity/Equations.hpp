// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
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
void primal_fluxes(
    gsl::not_null<tnsr::II<DataVector, Dim>*> flux_for_displacement,
    const tnsr::ii<DataVector, Dim>& strain,
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
 * \brief Compute the fluxes \f$F^i_{jk}=\delta^{i}_{(j} \xi_{k)}\f$ for the
 * auxiliary (strain) field in the first-order formulation of the Elasticity
 * equation.
 *
 * \see Elasticity::FirstOrderSystem
 */
template <size_t Dim>
void auxiliary_fluxes(
    gsl::not_null<tnsr::Ijj<DataVector, Dim>*> flux_for_strain,
    const tnsr::I<DataVector, Dim>& displacement);

/*!
 * \brief Compute the fluxes \f$F^i_{jk}=\delta^{i}_{(j}\gamma_{k)l}\xi^l\f$
 * for the auxiliary (strain) field in the first-order formulation of the
 * curved-space elasticity equations on a metric \f$\gamma_{ij}\f$.
 *
 * \see Elasticity::FirstOrderSystem
 */
template <size_t Dim>
void curved_auxiliary_fluxes(
    gsl::not_null<tnsr::Ijj<DataVector, Dim>*> flux_for_strain,
    const tnsr::ii<DataVector, Dim>& metric,
    const tnsr::I<DataVector, Dim>& displacement);

/*!
 * \brief Add the contribution \f$\Gamma_{ijk}\xi^i\f$ to the strain source for
 * the curved-space elasticity equations on a metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the divergence on a
 * curved background.
 */
template <size_t Dim>
void add_curved_auxiliary_sources(
    gsl::not_null<tnsr::ii<DataVector, Dim>*> source_for_strain,
    const tnsr::ijj<DataVector, Dim>& christoffel_first_kind,
    const tnsr::I<DataVector, Dim>& displacement);

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the Elasticity equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Elasticity::FirstOrderSystem
 */
template <size_t Dim>
struct Fluxes {
  using argument_tags =
      tmpl::list<Tags::ConstitutiveRelation<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using volume_tags = tmpl::list<Tags::ConstitutiveRelation<Dim>>;
  static void apply(
      gsl::not_null<tnsr::II<DataVector, Dim>*> flux_for_displacement,
      const ConstitutiveRelations::ConstitutiveRelation<Dim>&
          constitutive_relation,
      const tnsr::I<DataVector, Dim>& coordinates,
      const tnsr::ii<DataVector, Dim>& strain);
  static void apply(gsl::not_null<tnsr::Ijj<DataVector, Dim>*> flux_for_strain,
                    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
                        constitutive_relation,
                    const tnsr::I<DataVector, Dim>& coordinates,
                    const tnsr::I<DataVector, Dim>& displacement);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the Elasticity equation.
 *
 * \see Elasticity::FirstOrderSystem
 */
template <size_t Dim>
struct Sources {
  using argument_tags = tmpl::list<>;
  static void apply(
      gsl::not_null<tnsr::I<DataVector, Dim>*> equation_for_displacement,
      const tnsr::I<DataVector, Dim>& displacement,
      const tnsr::II<DataVector, Dim>& minus_stress);
  static void apply(
      gsl::not_null<tnsr::ii<DataVector, Dim>*> equation_for_strain,
      const tnsr::I<DataVector, Dim>& displacement);
};

}  // namespace Elasticity
