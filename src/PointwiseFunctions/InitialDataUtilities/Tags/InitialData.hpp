// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::initial_data {
/*!
 * \brief Option tags for initial data of evolution system.
 */
namespace OptionTags {
/*!
 * \ingroup OptionTagsGroup
 * \brief Class holding options for initial data of evolution system.
 */
struct InitialData {
  static constexpr Options::String help =
      "Options for initial data of evolution system";
  using type = std::unique_ptr<evolution::initial_data::InitialData>;
};
}  // namespace OptionTags

/*!
 * \brief Tags for initial data of evolution system.
 */
namespace Tags {
/*!
 * \brief The global cache tag for the initial data type
 */
struct InitialData : db::SimpleTag {
  using type = std::unique_ptr<evolution::initial_data::InitialData>;
  using option_tags =
      tmpl::list<evolution::initial_data::OptionTags::InitialData>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return deserialize<type>(serialize<type>(value).data());
  }
};
}  // namespace Tags
}  // namespace evolution::initial_data
