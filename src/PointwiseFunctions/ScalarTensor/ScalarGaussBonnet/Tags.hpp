// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingParameters.hpp"

/// \cond
namespace ScalarTensor::OptionTags {
struct Group;
}  // namespace ScalarTensor::OptionTags
/// \endcond

namespace ScalarTensor {
namespace OptionTags {
/*!
 * \brief Linear coupling parameters to curvature.
 */
struct CouplingParameters {
  static constexpr Options::String help = {"Coupling parameters to curvature."};
  using type = ScalarTensor::CouplingParameterOptions;
  using group = ScalarTensor::OptionTags::Group;
};

}  // namespace OptionTags

namespace Tags {
/*!
 * \brief Linear, quadratic and quartic coupling parameters to curvature.
 */
struct CouplingParameters : db::SimpleTag {
  using type = ScalarTensor::CouplingParameterOptions;
  using option_tags = tmpl::list<OptionTags::CouplingParameters>;
  static constexpr bool pass_metavariables = false;
  static ScalarTensor::CouplingParameterOptions create_from_options(
      const ScalarTensor::CouplingParameterOptions& coupling_parameters) {
    return coupling_parameters;
  }
};
}  // namespace Tags
}  // namespace ScalarTensor
