// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace Elasticity {
namespace ConstitutiveRelations {
template <size_t Dim>
struct IsotropicHomogeneous;  // IWYU pragma: keep
}  // namespace ConstitutiveRelations
}  // namespace Elasticity
/// \endcond

namespace Elasticity {
/*!
 * \brief Constitutive (stress-strain) relations that characterize the elastic
 * properties of a material
 */
namespace ConstitutiveRelations {

/*!
 * \brief Base class for constitutive (stress-strain) relations that
 * characterize the elastic properties of a material
 *
 * \details A constitutive relation, in the context of elasticity, relates the
 * Stress \f$T^{ij}\f$ and Strain \f$S_{ij}=\nabla_{(i}u_{j)}\f$ within an
 * elastic material (see \ref Elasticity). For small stresses it is approximated
 * by the linear relation \f[ T^{ij} = -Y^{ijkl}S_{kl} \f] (Eq. 11.17 in
 * \cite ThorneBlandford2017) that is referred to as _Hooke's law_. The
 * constitutive relation in this linear approximation is determined by the
 * elasticity (or _Young's_) tensor \f$Y^{ijkl}=Y^{(ij)(kl)}=Y^{klij}\f$ that
 * generalizes a simple proportionality to a three-dimensional and (possibly)
 * anisotropic material.
 *
 * \note We assume a Euclidean metric in Cartesian coordinates here (for now).
 */
template <size_t Dim>
class ConstitutiveRelation : public PUP::able {
 public:
  using creatable_classes = tmpl::list<IsotropicHomogeneous<Dim>>;

  ConstitutiveRelation() = default;
  ConstitutiveRelation(const ConstitutiveRelation&) = delete;
  ConstitutiveRelation& operator=(const ConstitutiveRelation&) = delete;
  ConstitutiveRelation(ConstitutiveRelation&&) = default;
  ConstitutiveRelation& operator=(ConstitutiveRelation&&) = default;
  ~ConstitutiveRelation() override = default;

  WRAPPED_PUPable_abstract(ConstitutiveRelation);  // NOLINT

  /// The constitutive relation that characterizes the elastic properties of a
  /// material
  virtual tnsr::II<DataVector, Dim> stress(
      const tnsr::ii<DataVector, Dim>& strain,
      const tnsr::I<DataVector, Dim>& x) const noexcept = 0;

  /// Symmmetrize the displacement gradient to compute the strain, then pass it
  /// to the constitutive relation
  tnsr::II<DataVector, Dim> stress(
      const tnsr::iJ<DataVector, Dim>& grad_displacement,
      const tnsr::I<DataVector, Dim>& x) const noexcept;
};

}  // namespace ConstitutiveRelations
}  // namespace Elasticity

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"  // IWYU pragma: keep
