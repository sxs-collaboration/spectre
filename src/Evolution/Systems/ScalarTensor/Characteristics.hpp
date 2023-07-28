// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::Tags {
struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

/*!
 * \brief Computes the largest magnitude of the characteristic speeds.
 *
 * \details Compute the maximum of the largest characteristic speed of each
 * component system, i.e. the largest speed between ::gh and ::CurvedScalarWave.
 */
template <typename Frame = Frame::Inertial>
struct ComputeLargestCharacteristicSpeed : db::ComputeTag,
                                           LargestCharacteristicSpeed {
  static constexpr size_t Dim = 3_st;
  using argument_tags =
      tmpl::list<::gh::ConstraintDamping::Tags::ConstraintGamma1,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, Dim, Frame>,
                 gr::Tags::SpatialMetric<DataVector, Dim, Frame>,
                 CurvedScalarWave::Tags::ConstraintGamma1>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(const gsl::not_null<double*> speed,
                       // GH arguments
                       const Scalar<DataVector>& gamma_1,
                       const Scalar<DataVector>& lapse,
                       const tnsr::I<DataVector, Dim, Frame>& shift,
                       const tnsr::ii<DataVector, Dim, Frame>& spatial_metric,
                       // Scalar arguments
                       const Scalar<DataVector>& gamma_1_scalar) {
    // Largest speed in for Generalized Harmonic
    double gh_largest_speed = 0.0;
    gh::Tags::ComputeLargestCharacteristicSpeed<Dim, Frame>::function(
        make_not_null(&gh_largest_speed), gamma_1, lapse, shift,
        spatial_metric);
    // Largest speed for CurvedScalarWave
    double scalar_largest_speed = 0.0;
    CurvedScalarWave::Tags::ComputeLargestCharacteristicSpeed<Dim>::function(
        make_not_null(&scalar_largest_speed), gamma_1_scalar, lapse, shift,
        spatial_metric);
    // Compute the maximum speed
    *speed = std::max(gh_largest_speed, scalar_largest_speed);
  }
};
}  // namespace ScalarTensor::Tags
