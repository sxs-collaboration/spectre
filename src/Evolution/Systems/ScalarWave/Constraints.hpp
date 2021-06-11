// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ScalarWave {
/// @{
/*!
 * \brief Compute the scalar-wave one-index constraint.
 *
 * \details Computes the scalar-wave one-index constraint,
 * \f$C_{i} = \partial_i\psi - \Phi_{i},\f$ which is
 * given by Eq. (19) of \cite Holst2004wt
 */
template <size_t SpatialDim>
tnsr::i<DataVector, SpatialDim, Frame::Inertial> one_index_constraint(
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept;

template <size_t SpatialDim>
void one_index_constraint(
    gsl::not_null<tnsr::i<DataVector, SpatialDim, Frame::Inertial>*> constraint,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& d_psi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept;
/// @}

/// @{
/*!
 * \brief Compute the scalar-wave 2-index constraint.
 *
 * \details Computes the scalar-wave 2-index constraint
 * \f$C_{ij} = \partial_i\Phi_j - \partial_j\Phi_i,\f$
 * where \f$\Phi_{i} = \partial_i\psi\f$, that is given by
 * Eq. (20) of \cite Holst2004wt
 *
 * \note We do not support custom storage for antisymmetric tensors yet.
 */
template <size_t SpatialDim>
tnsr::ij<DataVector, SpatialDim, Frame::Inertial> two_index_constraint(
    const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>& d_phi) noexcept;

template <size_t SpatialDim>
void two_index_constraint(
    gsl::not_null<tnsr::ij<DataVector, SpatialDim, Frame::Inertial>*>
        constraint,
    const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>& d_phi) noexcept;
/// @}

namespace Tags {
/*!
 * \brief Compute item to get the one-index constraint for the scalar-wave
 * evolution system.
 *
 * \details See `one_index_constraint()`. Can be retrieved using
 * `ScalarWave::Tags::OneIndexConstraint`.
 */
template <size_t SpatialDim>
struct OneIndexConstraintCompute : OneIndexConstraint<SpatialDim>,
                                   db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::deriv<Psi, tmpl::size_t<SpatialDim>, Frame::Inertial>,
                 Phi<SpatialDim>>;
  using return_type = tnsr::i<DataVector, SpatialDim, Frame::Inertial>;
  static constexpr void (*function)(
      const gsl::not_null<return_type*> result,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&,
      const tnsr::i<DataVector, SpatialDim, Frame::Inertial>&) =
      &one_index_constraint<SpatialDim>;
  using base = OneIndexConstraint<SpatialDim>;
};

/*!
 * \brief Compute item to get the two-index constraint for the scalar-wave
 * evolution system.
 *
 * \details See `two_index_constraint()`. Can be retrieved using
 * `ScalarWave::Tags::TwoIndexConstraint`.
 */
template <size_t SpatialDim>
struct TwoIndexConstraintCompute : TwoIndexConstraint<SpatialDim>,
                                   db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::deriv<Phi<SpatialDim>, tmpl::size_t<SpatialDim>,
                               Frame::Inertial>>;
  using return_type = tnsr::ij<DataVector, SpatialDim, Frame::Inertial>;
  static constexpr void (*function)(
      const gsl::not_null<return_type*> result,
      const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>&) =
      &two_index_constraint<SpatialDim>;
  using base = TwoIndexConstraint<SpatialDim>;
};

}  // namespace Tags
}  // namespace ScalarWave
