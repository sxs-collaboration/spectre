// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

namespace Ccz4::fd::Tags {
/// \brief Tags sent for second-order Ccz4 evolution.
using spacetime_reconstruction_tags = tmpl::list<
    ::Ccz4::Tags::ConformalMetric<DataVector, 3>, gr::Tags::Lapse<DataVector>,
    gr::Tags::Shift<DataVector, 3>, ::Ccz4::Tags::ConformalFactor<DataVector>,
    ::Ccz4::Tags::ATilde<DataVector, 3>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
    ::Ccz4::Tags::b<DataVector, 3>>;
}  // namespace Ccz4::fd::Tags
