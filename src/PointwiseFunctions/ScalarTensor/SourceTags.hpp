// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Options/String.hpp"

/// \cond
namespace ScalarTensor::OptionTags {
struct Group;
}  // namespace ScalarTensor::OptionTags
/// \endcond

namespace ScalarTensor {
namespace OptionTags {
/*!
 * \brief Start time for ramp up function.
 */
struct RampUpStart {
  static std::string name() { return "RampUpStart"; }
  using type = double;
  static constexpr Options::String help{"Start time for ramp up function"};
  using group = ::ScalarTensor::OptionTags::Group;
};

/*!
 * \brief Start time for ramp up function.
 */
struct RampUpDuration {
  static std::string name() { return "RampUpDuration"; }
  using type = double;
  static constexpr Options::String help{"Duration time for ramp up function"};
  using group = ::ScalarTensor::OptionTags::Group;
};

}  // namespace OptionTags

namespace Tags {
/*!
 * \brief Start and duration time for ramp up function.
 */
struct RampUpParameters : db::SimpleTag {
  using type = std::pair<double, double>;
  using option_tags =
      tmpl::list<OptionTags::RampUpStart, OptionTags::RampUpDuration>;
  static constexpr bool pass_metavariables = false;
  static std::pair<double, double> create_from_options(
      const double start_time, const double duration_time) {
    return std::pair<double, double> {start_time, duration_time};
  }
};

}  // namespace Tags
}  // namespace ScalarTensor
