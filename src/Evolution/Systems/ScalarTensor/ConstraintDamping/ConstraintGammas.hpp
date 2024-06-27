// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace ScalarTensor::ConstraintDamping::Tags {
/*!
 * \brief Computes the constraint damping parameter \f$\gamma_1\f$ from the
 * coordinates and a DampingFunction.
 *
 * \details Can be retrieved using
 * `gh::ConstraintDamping::Tags::ConstraintGamma1`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma1Compute : ::CurvedScalarWave::Tags::ConstraintGamma1,
                                 db::ComputeTag {
  using argument_tags =
      tmpl::list<DampingFunctionGamma1<SpatialDim, Frame>,
                 domain::Tags::Coordinates<SpatialDim, Frame>, ::Tags::Time,
                 ::domain::Tags::FunctionsOfTime>;
  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma1,
      const ::ScalarTensor::ConstraintDamping::DampingFunction<
          SpatialDim, Frame>& damping_function,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords, const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    damping_function(gamma1, coords, time, functions_of_time);
  }

  using base = ::CurvedScalarWave::Tags::ConstraintGamma1;
};

/*!
 * \brief Computes the constraint damping parameter \f$\gamma_2\f$ from the
 * coordinates and a DampingFunction.
 *
 * \details Can be retrieved using
 * `gh::ConstraintDamping::Tags::ConstraintGamma2`.
 */
template <size_t SpatialDim, typename Frame>
struct ConstraintGamma2Compute : ::CurvedScalarWave::Tags::ConstraintGamma2,
                                 db::ComputeTag {
  using argument_tags =
      tmpl::list<DampingFunctionGamma2<SpatialDim, Frame>,
                 domain::Tags::Coordinates<SpatialDim, Frame>, ::Tags::Time,
                 ::domain::Tags::FunctionsOfTime>;
  using return_type = Scalar<DataVector>;

  static constexpr void function(
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const ::ScalarTensor::ConstraintDamping::DampingFunction<
          SpatialDim, Frame>& damping_function,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords, const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    damping_function(gamma, coords, time, functions_of_time);
  }

  using base = ::CurvedScalarWave::Tags::ConstraintGamma2;
};
}  // namespace ScalarTensor::ConstraintDamping::Tags
