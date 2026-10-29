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
#include "Utilities/TMPL.hpp"

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
void potential_energy_density(gsl::not_null<Scalar<DataVector>*> result,
                              const tnsr::ii<DataVector, Dim>& strain,
                              const tnsr::II<DataVector, Dim>& stress);

template <size_t Dim>
void potential_energy_density(
    gsl::not_null<Scalar<DataVector>*> result,
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

/*!
 * \brief The bulk (volumetric) part of the potential energy density
 * \f$U_\mathrm{bulk}=-\frac{1}{2 \mathrm{Dim}} \Theta \Sigma\f$, where
 * \f$\Theta = S_{ii}\f$ is the trace of the strain and
 * \f$\Sigma = T^{ii}\f$ is the trace of the stress.
 *
 * The traces run over the Dim dimensions of the domain. Only for Dim = 3 is
 * this the physical volumetric/deviatoric split. In 2D (plane stress) the
 * out-of-plane strain \f$S_{zz}\f$ is generally nonzero but not included, so
 * the split is only the 2D trace decomposition.
 *
 * \see `Elasticity::Tags::BulkPotentialEnergyDensity`
 */
template <size_t Dim>
void bulk_potential_energy_density(gsl::not_null<Scalar<DataVector>*> result,
                                   const tnsr::ii<DataVector, Dim>& strain,
                                   const tnsr::II<DataVector, Dim>& stress);

/*!
 * \brief The shear (deviatoric) part of the potential energy density
 * \f$U_\mathrm{shear} = -\frac{1}{2} s^{ij} e_{ij}\f$, computed directly from
 * the deviatoric strain \f$e_{ij} = S_{ij} - \frac{1}{\mathrm{Dim}} \delta_{ij}
 * \Theta\f$ and deviatoric stress \f$s^{ij} = T^{ij} - \frac{1}{\mathrm{Dim}}
 * \delta^{ij} \Sigma\f$.
 *
 * Bulk + shear = total (`potential_energy_density`) is an algebraic identity
 * for any symmetric strain and stress. The shear part is computed
 * independently of the total (not as total minus bulk), so checking the
 * identity numerically tests the implementation. The traces run over the Dim
 * dimensions of the domain; see `bulk_potential_energy_density` for the
 * meaning in 2D.
 *
 * \see `Elasticity::Tags::ShearPotentialEnergyDensity`
 */
template <size_t Dim>
void shear_potential_energy_density(gsl::not_null<Scalar<DataVector>*> result,
                                    const tnsr::ii<DataVector, Dim>& strain,
                                    const tnsr::II<DataVector, Dim>& stress);

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
      tmpl::list<Elasticity::Tags::Strain<Dim>, Elasticity::Tags::Stress<Dim>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, Dim>&,
      const tnsr::II<DataVector, Dim>&)>(&potential_energy_density<Dim>);
};

/// \brief Computes the bulk (volumetric) part of the potential energy density,
/// \f$U_\mathrm{bulk}=-\frac{1}{2 \mathrm{Dim}} \Theta \Sigma\f$, from the
/// strain and the stress. The traces run over the Dim dimensions of the
/// domain, so this is the physical volumetric part only in 3D.
/// \see `Elasticity::Tags::BulkPotentialEnergyDensity`
template <size_t Dim>
struct BulkPotentialEnergyDensityCompute
    : Elasticity::Tags::BulkPotentialEnergyDensity<Dim>,
      db::ComputeTag {
  using base = Elasticity::Tags::BulkPotentialEnergyDensity<Dim>;
  using return_type = Scalar<DataVector>;
  using argument_tags =
      tmpl::list<Elasticity::Tags::Strain<Dim>, Elasticity::Tags::Stress<Dim>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, Dim>&,
      const tnsr::II<DataVector, Dim>&)>(&bulk_potential_energy_density<Dim>);
};

/// \brief Computes the shear (deviatoric) part of the potential energy
/// density, \f$U_\mathrm{shear} = -\frac{1}{2} s^{ij} e_{ij}\f$, from the
/// deviators of the strain and the stress. The traces run over the Dim
/// dimensions of the domain, so this is the physical deviatoric part only in
/// 3D.
/// \see `Elasticity::Tags::ShearPotentialEnergyDensity`
template <size_t Dim>
struct ShearPotentialEnergyDensityCompute
    : Elasticity::Tags::ShearPotentialEnergyDensity<Dim>,
      db::ComputeTag {
  using base = Elasticity::Tags::ShearPotentialEnergyDensity<Dim>;
  using return_type = Scalar<DataVector>;
  using argument_tags =
      tmpl::list<Elasticity::Tags::Strain<Dim>, Elasticity::Tags::Stress<Dim>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, Dim>&,
      const tnsr::II<DataVector, Dim>&)>(&shear_potential_energy_density<Dim>);
};

/// The bulk and shear energy density compute tags that an executable should
/// observe: both in 3D, none in 2D. The bulk/shear split uses the
/// Dim-dimensional trace, which is the physical volumetric/deviatoric
/// decomposition only in 3D (in 2D plane stress the out-of-plane strain
/// \f$S_{zz}\f$ is generally nonzero but not part of the 2D strain).
template <size_t Dim>
using bulk_shear_observables =
    tmpl::conditional_t<Dim == 3,
                        tmpl::list<BulkPotentialEnergyDensityCompute<Dim>,
                                   ShearPotentialEnergyDensityCompute<Dim>>,
                        tmpl::list<>>;

}  // namespace Tags
}  // namespace Elasticity
