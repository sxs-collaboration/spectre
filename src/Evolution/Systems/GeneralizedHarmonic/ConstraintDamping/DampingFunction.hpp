// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
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
  DampingFunction(const DampingFunction& /*rhs*/) = default;
  DampingFunction& operator=(const DampingFunction& /*rhs*/) = default;
  DampingFunction(DampingFunction&& /*rhs*/) noexcept = default;
  DampingFunction& operator=(DampingFunction&& /*rhs*/) noexcept = default;
  ~DampingFunction() override = default;

  explicit DampingFunction(CkMigrateMessage* msg) noexcept : PUP::able(msg) {}

  //@{
  /// Returns the value of the function at the coordinate 'x'.
  virtual void operator()(
      const gsl::not_null<Scalar<double>*> value_at_x,
      const tnsr::I<double, VolumeDim, Fr>& x, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept = 0;
  virtual void operator()(
      const gsl::not_null<Scalar<DataVector>*> value_at_x,
      const tnsr::I<DataVector, VolumeDim, Fr>& x, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept = 0;
  //@}

  virtual auto get_clone() const noexcept
      -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> = 0;
};
}  // namespace GeneralizedHarmonic::ConstraintDamping

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/GaussianPlusConstant.hpp"
