// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DhGaugeParameters.hpp"
#include "Options/Options.hpp"

/// \cond
namespace GeneralizedHarmonic::OptionTags {
struct Group;
}  // namespace GeneralizedHarmonic::OptionTags
/// \endcond

namespace GeneralizedHarmonic::gauges {
namespace OptionTags {
template <bool UseRollon>
struct DhGaugeParameters {
  using type = GeneralizedHarmonic::gauges::DhGaugeParameters<UseRollon>;
  static constexpr Options::String help{
      "Parameters for initializing damped harmonic gauge."};
  using group = GeneralizedHarmonic::OptionTags::Group;
};
}  // namespace OptionTags

namespace Tags {
/// \brief Input option tags for the generalized harmonic evolution system
template <bool UseRollon>
struct DhGaugeParameters : db::SimpleTag {
  using ParametersType =
      GeneralizedHarmonic::gauges::DhGaugeParameters<UseRollon>;
  using type = ParametersType;
  using option_tags = tmpl::list<
      GeneralizedHarmonic::gauges::OptionTags::DhGaugeParameters<UseRollon>>;

  static constexpr bool pass_metavariables = false;
  static ParametersType create_from_options(const ParametersType& parameters) {
    return parameters;
  }
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic::gauges
