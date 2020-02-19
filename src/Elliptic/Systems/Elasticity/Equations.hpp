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
 * \brief Compute the fluxes \f$F^{ij}=Y^{ijkl}(x) S_{kl}(x)\f$ for
 * the Elasticity equation on a flat spatial metric in Cartesian coordinates.
 */
template <size_t Dim>
void primal_fluxes(
    gsl::not_null<tnsr::IJ<DataVector, Dim>*> flux_for_displacement,
    const tnsr::ii<DataVector, Dim>& strain,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation,
    const tnsr::I<DataVector, Dim>& coordinates) noexcept;

/*!
 * \brief Compute the fluxes \f$F^i_{jk}=\delta^{i}_{(j} \xi_{k)}\f$ for the
 * auxiliary field in the first-order formulation of the Elasticity equation.
 *
 * \see Elasticity::FirstOrderSystem
 */
template <size_t Dim>
void auxiliary_fluxes(
    gsl::not_null<tnsr::Ijj<DataVector, Dim>*> flux_for_strain,
    const tnsr::I<DataVector, Dim>& displacement) noexcept;

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the Elasticity equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Elasticity::FirstOrderSystem
 */
template <size_t Dim>
struct Fluxes {
  using argument_tags =
      tmpl::list<Tags::ConstitutiveRelationBase,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<tnsr::IJ<DataVector, Dim>*> flux_for_displacement,
      const ConstitutiveRelations::ConstitutiveRelation<Dim>&
          constitutive_relation,
      const tnsr::I<DataVector, Dim>& coordinates,
      const tnsr::ii<DataVector, Dim>& strain) noexcept {
    primal_fluxes(flux_for_displacement, strain, constitutive_relation,
                  coordinates);
  }
  static void apply(
      const gsl::not_null<tnsr::Ijj<DataVector, Dim>*> flux_for_strain,
      const ConstitutiveRelations::ConstitutiveRelation<
          Dim>& /*constitutive_relation*/,
      const tnsr::I<DataVector, Dim>& /*coordinates*/,
      const tnsr::I<DataVector, Dim>& displacement) noexcept {
    auxiliary_fluxes(flux_for_strain, displacement);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

/*!
 * \brief Compute the sources \f$S_A\f$ for the Elasticity equation.
 *
 * \see Elasticity::FirstOrderSystem
 */
template <size_t Dim>
struct Sources {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> source_for_displacement,
      const tnsr::I<DataVector, Dim>& /*displacement*/) noexcept {
    for (size_t d = 0; d < Dim; d++) {
      source_for_displacement->get(d) = 0.;
    }
  }
};

}  // namespace Elasticity
