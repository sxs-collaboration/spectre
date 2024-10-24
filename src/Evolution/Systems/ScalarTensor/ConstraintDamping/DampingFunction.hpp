// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
class DataVector;
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

/// Holds classes implementing DampingFunction (functions \f$R^n \to R\f$).
namespace ScalarTensor::ConstraintDamping {
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
  using creatable_classes = tmpl::conditional_t<
      (VolumeDim == 3 and std::is_same<Fr, Frame::Grid>::value),
      tmpl::list<
          ScalarTensor::ConstraintDamping::GaussianPlusConstant<VolumeDim, Fr>>,
      tmpl::list<ScalarTensor::ConstraintDamping::GaussianPlusConstant<
          VolumeDim, Fr>>>;
  constexpr static size_t volume_dim = VolumeDim;
  using frame = Fr;

  WRAPPED_PUPable_abstract(DampingFunction);  // NOLINT

  DampingFunction() = default;
  DampingFunction(const DampingFunction& /*rhs*/) = default;
  DampingFunction& operator=(const DampingFunction& /*rhs*/) = default;
  DampingFunction(DampingFunction&& /*rhs*/) = default;
  DampingFunction& operator=(DampingFunction&& /*rhs*/) = default;
  ~DampingFunction() override = default;

  explicit DampingFunction(CkMigrateMessage* msg) : PUP::able(msg) {}

  /// @{
  /// Returns the value of the function at the coordinate 'x'.
  virtual void operator()(
      gsl::not_null<Scalar<double>*> value_at_x,
      const tnsr::I<double, VolumeDim, Fr>& x, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const = 0;
  virtual void operator()(
      gsl::not_null<Scalar<DataVector>*> value_at_x,
      const tnsr::I<DataVector, VolumeDim, Fr>& x, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const = 0;
  /// @}

  virtual auto get_clone() const
      -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> = 0;
};
}  // namespace ScalarTensor::ConstraintDamping

#include "Evolution/Systems/ScalarTensor/ConstraintDamping/GaussianPlusConstant.hpp"
