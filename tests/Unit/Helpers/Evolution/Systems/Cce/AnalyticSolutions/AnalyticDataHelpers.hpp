// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Solutions {
namespace TestHelpers {

// This function determines the Bondi-Sachs scalars from a Cartesian spacetime
// metric, assuming that the metric is already in null form, so the spatial
// coordinates are related to standard Bondi-Sachs coordinates by just the
// standard Cartesian to spherical Jacobian.
tuples::TaggedTuple<Tags::BondiBeta, Tags::BondiU, Tags::BondiW, Tags::BondiJ>
extract_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    double extraction_radius) noexcept;

// This function determines the time derivative of the Bondi-Sachs scalars
// from the time derivative of a Cartesian spacetime metric, the Cartesian
// metric, and Jacobian factors. This procedure assumes that the metric is
// already in null form, so the spatial coordinates are related to standard
// Bondi-Sachs coordinates by just the standard cartesian to spherical Jacobian.
tuples::TaggedTuple<::Tags::dt<Tags::BondiBeta>, ::Tags::dt<Tags::BondiU>,
                    ::Tags::dt<Tags::BondiW>, ::Tags::dt<Tags::BondiJ>>
extract_dt_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    double extraction_radius) noexcept;

// This function determines the radial derivative of the Bondi-Sachs scalars
// from the radial derivative of a Cartesian spacetime metric, the Cartesian
// metric, and Jacobian factors. This procedure assumes that the metric is
// already in null form, so the spatial coordinates are related to standard
// Bondi-Sachs coordinates by just the standard cartesian to spherical Jacobian.
tuples::TaggedTuple<Tags::Dr<Tags::BondiBeta>, Tags::Dr<Tags::BondiU>,
                    Tags::Dr<Tags::BondiW>, Tags::Dr<Tags::BondiJ>>
extract_dr_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& dr_spacetime_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    const CartesianiSphericalJ& dr_inverse_jacobian,
    double extraction_radius) noexcept;
}  // namespace TestHelpers
}  // namespace Solutions
}  // namespace Cce
