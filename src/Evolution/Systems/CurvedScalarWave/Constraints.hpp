// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to calculate the scalar wave constraints in
/// curved spacetime

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave {
namespace Tags {
/*!
 * \brief Compute items to compute constraint damping parameters.
 *
 * \details For the evolution system with constraint damping parameters
 * to be symmetric-hyperbolic, we need \f$\gamma_1 \gamma_2 = 0\f$. When
 * \f$\gamma_1 = 0\f$, Ref. \cite Holst2004wt shows that the one-index
 * constraint decays exponentially on a time-scale \f$ 1/\gamma_2\f$.
 * Conversely, they also show that using \f$\gamma_2 > 0\f$ leads to
 * exponential suppression of constraint violations.
 *
 * Can be retrieved using `CurvedScalarWave::Tags::ConstraintGamma1`
 * and `CurvedScalarWave::Tags::ConstraintGamma2`.
 */
struct ConstraintGamma1Compute : ConstraintGamma1, db::ComputeTag {
  using argument_tags = tmpl::list<Psi>;
  static auto function(const Scalar<DataVector>& used_for_size) noexcept {
    return make_with_value<type>(used_for_size, 0.);
  }
  using base = ConstraintGamma1;
};
/// \copydoc ConstraintGamma1Compute
struct ConstraintGamma2Compute : ConstraintGamma2, db::ComputeTag {
  using argument_tags = tmpl::list<Psi>;
  static auto function(const Scalar<DataVector>& used_for_size) noexcept {
    return make_with_value<type>(used_for_size, 1.);
  }
  using base = ConstraintGamma2;
};
}  // namespace Tags

// @{
/*!
 * \brief Computes the scalar-wave one-index constraint.
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
// @}

// @{
/*!
 * \brief Computes the scalar-wave 2-index constraint.
 *
 * \details Computes the scalar-wave 2-index FOSH constraint
 * [Eq. (20) of \cite Holst2004wt],
 *
 * \f{eqnarray}{
 * C_{ij} &\equiv& \partial_i \Phi_j - \partial_j \Phi_i
 * \f}
 *
 * where \f$\Phi_{i} = \partial_i\psi\f$; and \f$\psi\f$ is the scalar field.
 */
template <size_t SpatialDim>
tnsr::ij<DataVector, SpatialDim, Frame::Inertial> two_index_constraint(
    const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>& d_phi) noexcept;

template <size_t SpatialDim>
void two_index_constraint(
    gsl::not_null<tnsr::ij<DataVector, SpatialDim, Frame::Inertial>*>
        constraint,
    const tnsr::ij<DataVector, SpatialDim, Frame::Inertial>& d_phi) noexcept;
// @}

namespace Tags {
/*!
 * \brief Compute item to get the one-index constraint for the scalar-wave
 * evolution system.
 *
 * \details See `one_index_constraint()`. Can be retrieved using
 * `CurvedScalarWave::Tags::OneIndexConstraint`.
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
 * `CurvedScalarWave::Tags::TwoIndexConstraint`.
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
}  // namespace CurvedScalarWave
