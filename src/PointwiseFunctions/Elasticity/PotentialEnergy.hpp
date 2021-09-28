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

/// @{
/*!
 * \brief The potential energy density \f$U=-\frac{1}{2}S_{ij}T^{ij}\f$ stored
 * in the deformation of the elastic material (see Eq. (11.25) in
 * \cite ThorneBlandford2017)
 *
 * Note that the two-dimensional instantiation of this function assumes that
 * only the terms of \f$S_{ij}T^{ij}\f$ where both \f$i\f$ and \f$j\f$
 * correspond to one of the computational dimensions contribute to the sum. This
 * is the case for the plane-stress approximation employed in the
 * two-dimensional `Elasticity::ConstitutiveRelations::IsotropicHomogeneous`,
 * for example, where \f$T^{i3}=0=T^{3i}\f$.
 */
template <size_t Dim>
void potential_energy_density(
    gsl::not_null<Scalar<DataVector>*> potential_energy_density,
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::I<DataVector, Dim>& coordinates,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation);

template <size_t Dim>
Scalar<DataVector> potential_energy_density(
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::I<DataVector, Dim>& coordinates,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation);
/// @}

namespace Tags {

/// \brief Computes the energy density stored in the deformation of the elastic
/// material.
/// \see `Elasticity::Tags::PotentialEnergyDensity`
template <size_t Dim>
struct PotentialEnergyDensityCompute
    : Elasticity::Tags::PotentialEnergyDensity<Dim>,
      db::ComputeTag {
  using base = Elasticity::Tags::PotentialEnergyDensity<Dim>;
  using return_type = Scalar<DataVector>;
  using argument_tags =
      tmpl::list<Elasticity::Tags::Strain<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 Elasticity::Tags::ConstitutiveRelation<Dim>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, Dim>&,
      const tnsr::I<DataVector, Dim>&,
      const ConstitutiveRelations::ConstitutiveRelation<Dim>&)>(
      &potential_energy_density<Dim>);
};

}  // namespace Tags
}  // namespace Elasticity
