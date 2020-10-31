// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"

/// \cond
class DataVector;
/// \endcond

/// Holds classes implementing DampingFunction (functions \f$R^n \to R\f$).
namespace GeneralizedHarmonic::ConstraintDamping {
/// \cond
template <size_t VolumeDim, typename Fr>
class GaussianPlusConstant;
/// \endcond

/*!
 * \brief Base class defining interface for constraint damping functions.
 *
 * Encodes a function \f$R^n \to R\f$ where n is `VolumeDim` that represents
 * a generalized-harmonic constraint-damping parameter (i.e., Gamma0,
 * Gamma1, or Gamma2).
 */
template <size_t VolumeDim, typename Fr>
class DampingFunction : public PUP::able {
 public:
  using creatable_classes =
      tmpl::list<GeneralizedHarmonic::ConstraintDamping::GaussianPlusConstant<
          VolumeDim, Fr>>;
  constexpr static size_t volume_dim = VolumeDim;
  using frame = Fr;

  WRAPPED_PUPable_abstract(DampingFunction);  // NOLINT

  DampingFunction() = default;
  DampingFunction(const DampingFunction& /*rhs*/) = delete;
  DampingFunction& operator=(const DampingFunction& /*rhs*/) = delete;
  DampingFunction(DampingFunction&& /*rhs*/) noexcept = default;
  DampingFunction& operator=(DampingFunction&& /*rhs*/) noexcept = default;
  ~DampingFunction() override = default;

  //@{
  /// Returns the value of the function at the coordinate 'x'.
  virtual Scalar<double> operator()(
      const tnsr::I<double, VolumeDim, Fr>& x) const noexcept = 0;
  virtual Scalar<DataVector> operator()(
      const tnsr::I<DataVector, VolumeDim, Fr>& x) const noexcept = 0;
  //@}
};
}  // namespace GeneralizedHarmonic::ConstraintDamping

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/GaussianPlusConstant.hpp"
