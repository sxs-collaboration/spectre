// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
struct System {
  using variables_tag = ::Tags::Variables<tmpl::list<
      Tags::ConformalMetric<DataVector, 3>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, 3>, Tags::ConformalFactor<DataVector>,
      Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>, Tags::Theta<DataVector>,
      Tags::GammaHat<DataVector, 3>, Tags::AuxiliaryShiftB<DataVector, 3>>>;

  using gradients_tags =
      tmpl::list<Tags::ConformalMetric<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 Tags::ConformalFactor<DataVector>, Tags::ATilde<DataVector, 3>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 Tags::Theta<DataVector>, Tags::GammaHat<DataVector, 3>,
                 Tags::AuxiliaryShiftB<DataVector, 3>>;
};
}  // namespace Ccz4::fd
