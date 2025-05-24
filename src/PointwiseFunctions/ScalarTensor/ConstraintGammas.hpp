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
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/ScalarTensor/ConstraintDampingTags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace ScalarTensor::Tags {
/*!
 * \brief Computes the constraint damping parameter \f$\gamma_1\f$ from the
 * coordinates and a DampingFunction.
 *
 * \details Can be retrieved using
 * `CurvedScalarWave::Tags::ConstraintGamma1`.
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
      const gsl::not_null<Scalar<DataVector>*> gamma,
      const ::ConstraintDamping::DampingFunction<SpatialDim, Frame>&
          damping_function,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords, const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    damping_function(gamma, coords, time, functions_of_time);
  }

  using base = ::CurvedScalarWave::Tags::ConstraintGamma1;
};

/*!
 * \brief Computes the constraint damping parameter \f$\gamma_2\f$ from the
 * coordinates and a DampingFunction.
 *
 * \details Can be retrieved using
 * `CurvedScalarWave::::Tags::ConstraintGamma2`.
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
      const ::ConstraintDamping::DampingFunction<SpatialDim, Frame>&
          damping_function,
      const tnsr::I<DataVector, SpatialDim, Frame>& coords, const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) {
    damping_function(gamma, coords, time, functions_of_time);
  }

  using base = ::CurvedScalarWave::Tags::ConstraintGamma2;
};
}  // namespace ScalarTensor::Tags
