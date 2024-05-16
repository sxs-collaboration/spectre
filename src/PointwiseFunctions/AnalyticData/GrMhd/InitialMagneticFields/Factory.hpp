// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/InitialMagneticField.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Poloidal.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/InitialMagneticFields/Toroidal.hpp"

namespace grmhd::AnalyticData::InitialMagneticFields {

/// Typelist of available InitialMagneticFields
using initial_magnetic_fields = tmpl::list<Poloidal, Toroidal>;

}  // namespace grmhd::AnalyticData::InitialMagneticFields
