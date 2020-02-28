// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"

namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Holds the `OptionTags::AnalyticData` option in the input file
struct AnalyticDataGroup {
  static std::string name() noexcept { return "AnalyticData"; }
  static constexpr OptionString help =
      "Analytic data used for the initial data";
};

/// \ingroup OptionTagsGroup
/// The analytic data, with the type of the analytic data set as the template
/// parameter
template <typename DataType>
struct AnalyticData {
  static std::string name() noexcept { return option_name<DataType>(); }
  static constexpr OptionString help = "Options for the analytic data";
  using type = DataType;
  using group = AnalyticDataGroup;
};
}  // namespace OptionTags

namespace Tags {
struct AnalyticSolutionOrData : db::BaseTag {};

/// Can be used to retrieve the analytic solution from the cache without having
/// to know the template parameters of AnalyticData.
struct AnalyticDataBase : AnalyticSolutionOrData {};

/// The analytic data, with the type of the analytic data set as the
/// template parameter
template <typename DataType>
struct AnalyticData : AnalyticDataBase, db::SimpleTag {
  using type = DataType;
  using option_tags = tmpl::list<::OptionTags::AnalyticData<DataType>>;

  static constexpr bool pass_metavariables = false;
  static DataType create_from_options(
      const DataType& analytic_solution) noexcept {
    return deserialize<type>(serialize<type>(analytic_solution).data());
  }
};
}  // namespace Tags
