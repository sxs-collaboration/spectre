// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

namespace InitialDataUtilities {
namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * Holds the initial data option in the input file
 */
template <typename InitialDataType>
struct InitialData {
  static constexpr Options::String help =
      "Options for initial data of evolution system";
  using type = std::unique_ptr<InitialDataType>;
};
}  // namespace OptionTags

namespace Tags {
// Base tag for the InitialData struct. Can be used to retrieve the InitialData
// type object from the cache without having to know its template parameter.
struct InitialDataBase : db::BaseTag {};

/*!
 * \brief The global cache tag for the initial data type
 */
template <typename InitialDataType>
struct InitialData : InitialDataBase, db::SimpleTag {
  using type = std::unique_ptr<InitialDataType>;
  using option_tags = tmpl::list<
      InitialDataUtilities::OptionTags::InitialData<InitialDataType>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return deserialize<type>(serialize<type>(value).data());
  }
};
}  // namespace Tags
}  // namespace InitialDataUtilities
