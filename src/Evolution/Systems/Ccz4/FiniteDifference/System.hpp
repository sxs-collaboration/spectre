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
  // The free parameter f in the Gamma-driver condition.
  static constexpr double f = 0.75;
  // Whether to add the advective terms in the Gamma-driver condition,
  // i.e. in time derivatives of the shift and the auxiliary field b.
  static constexpr bool shifting_shift = false;

  using variables_tag = ::Tags::Variables<tmpl::list<
      Tags::ConformalMetric<DataVector, 3>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, 3>, Tags::ConformalFactor<DataVector>,
      Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>, Tags::Theta<DataVector>,
      Tags::GammaHat<DataVector, 3>, Tags::AuxiliaryShiftB<DataVector, 3>>>;

  using variables_tag_list = typename variables_tag::tags_list;

  using gradients_tags = variables_tag_list;
};
}  // namespace Ccz4::fd
