// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Holds the `OptionTags::AnalyticData` option in the input file
struct AnalyticDataGroup {
  static std::string name() { return "AnalyticData"; }
  static constexpr Options::String help =
      "Analytic data used for the initial data";
};

/// \ingroup OptionTagsGroup
/// The analytic data, with the type of the analytic data set as the template
/// parameter
template <typename DataType>
struct AnalyticData {
  static std::string name() { return pretty_type::name<DataType>(); }
  static constexpr Options::String help = "Options for the analytic data";
  using type = DataType;
  using group = AnalyticDataGroup;
};
}  // namespace OptionTags

namespace Tags {
/// The analytic data, with the type of the analytic data set as the
/// template parameter
template <typename DataType>
struct AnalyticData : db::SimpleTag {
  using type = DataType;
  using option_tags = tmpl::list<::OptionTags::AnalyticData<DataType>>;

  static constexpr bool pass_metavariables = false;
  static DataType create_from_options(const DataType& analytic_solution) {
    return serialize_and_deserialize<type>(analytic_solution);
  }
};
}  // namespace Tags
