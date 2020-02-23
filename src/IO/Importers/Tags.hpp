// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/ElementIndex.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Options/Options.hpp"

/// Items related to loading data from files
namespace importers {

/// The input file options associated with the data importer
namespace OptionTags {

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups the data importer configurations in the input file
 */
struct Group {
  static std::string name() noexcept { return "Importers"; }
  static constexpr OptionString help = "Options for loading data files";
};

/*!
 * \brief The file to read data from.
 */
template <typename ImporterOptionsGroup>
struct FileName {
  static_assert(
      cpp17::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'Importers' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = std::string;
  static constexpr OptionString help = "Path to the data file";
  using group = ImporterOptionsGroup;
};

/*!
 * \brief The subgroup within the file to read data from.
 *
 * This subgroup should conform to the `h5::VolumeData` format.
 */
template <typename ImporterOptionsGroup>
struct Subgroup {
  static_assert(
      cpp17::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'Importers' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = std::string;
  static constexpr OptionString help =
      "The subgroup within the file, excluding extensions";
  using group = ImporterOptionsGroup;
};

/*!
 * \brief The observation value at which to read data from the file.
 */
template <typename ImporterOptionsGroup>
struct ObservationValue {
  static_assert(
      cpp17::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'Importers' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = double;
  static constexpr OptionString help =
      "The observation value at which to read data";
  using group = ImporterOptionsGroup;
};

}  // namespace OptionTags

/// The \ref DataBoxGroup tags associated with the data importer
namespace Tags {

/*!
 * \brief The file to read data from.
 */
template <typename ImporterOptionsGroup>
struct FileName : db::SimpleTag {
  static std::string name() noexcept {
    return "FileName(" + option_name<ImporterOptionsGroup>() + ")";
  }
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::FileName<ImporterOptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& file_name) noexcept {
    return file_name;
  }
};

/*!
 * \brief The subgroup within the file to read data from.
 *
 * This subgroup should conform to the `h5::VolumeData` format.
 */
template <typename ImporterOptionsGroup>
struct Subgroup : db::SimpleTag {
  static std::string name() noexcept {
    return "Subgroup(" + option_name<ImporterOptionsGroup>() + ")";
  }
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::Subgroup<ImporterOptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& subgroup) noexcept {
    return subgroup;
  }
};

/*!
 * \brief The observation value at which to read data from the file.
 */
template <typename ImporterOptionsGroup>
struct ObservationValue : db::SimpleTag {
  static std::string name() noexcept {
    return "ObservationValue(" + option_name<ImporterOptionsGroup>() + ")";
  }
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::ObservationValue<ImporterOptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& observation_value) noexcept {
    return observation_value;
  }
};

/*!
 * \brief The elements that will receive data from the importer.
 *
 * \details Identifiers for elements from multiple parallel components can be
 * stored. Each element is identified by an `observers::ArrayComponentId` and
 * also needs to provide the `std::string` that identifies it in the data file.
 */
struct RegisteredElements : db::SimpleTag {
  using type = std::unordered_map<observers::ArrayComponentId, std::string>;
};

}  // namespace Tags

}  // namespace importers
