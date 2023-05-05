// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::Tags {
struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

/*!
 * \brief Computes the largest magnitude of the characteristic speeds.
 *
 * \warning Assumes \f$-1\le\gamma_1\le0\f$.
 */
template <typename Frame = Frame::Inertial>
struct ComputeLargestCharacteristicSpeed : db::ComputeTag,
                                           LargestCharacteristicSpeed {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, 3, Frame>,
                 gr::Tags::SpatialMetric<DataVector, 3, Frame>>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(const gsl::not_null<double*> speed,
                       const Scalar<DataVector>& lapse,
                       const tnsr::I<DataVector, 3, Frame>& shift,
                       const tnsr::ii<DataVector, 3, Frame>& spatial_metric) {
    const auto shift_magnitude = magnitude(shift, spatial_metric);
    *speed = max(get(shift_magnitude) + get(lapse));
  }
};
}  // namespace grmhd::GhValenciaDivClean::Tags
