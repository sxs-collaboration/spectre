// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/ElementIndex.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Options/Options.hpp"

/// Items related to loading volume data from a file and distributing it to
/// elements of an array component.
namespace importer {

/// The input file options associated with the data importer
namespace OptionTags {

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups the data importer configurations in the input file
 */
struct Group {
  static std::string name() noexcept { return "DataImporters"; }
  static constexpr OptionString help = "Options for loading volume data files";
};

/*!
 * \brief The file to read volume data from.
 */
template <typename ImporterOptionsGroup>
struct DataFileName {
  static_assert(
      cpp17::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'DataImporters' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = std::string;
  static constexpr OptionString help = "Path to the data file";
  using group = ImporterOptionsGroup;
};

/*!
 * \brief The subgroup within the data file to read volume data from.
 *
 * This subgroup should conform to the `h5::VolumeData` format.
 */
template <typename ImporterOptionsGroup>
struct VolumeDataSubgroup {
  static_assert(
      cpp17::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'DataImporters' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = std::string;
  static constexpr OptionString help =
      "Name of the subgroup within the file, excluding '.vol'";
  using group = ImporterOptionsGroup;
};

/*!
 * \brief The observation value at which to read volume data from the file.
 */
template <typename ImporterOptionsGroup>
struct ObservationValue {
  static_assert(
      cpp17::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'DataImporters' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = double;
  static constexpr OptionString help =
      "The observation value at which to read volume data";
  using group = ImporterOptionsGroup;
};

}  // namespace OptionTags

/// The \ref DataBoxGroup tags associated with the data importer
namespace Tags {

/*!
 * \brief The file to read volume data from.
 */
template <typename ImporterOptionsGroup>
struct DataFileName : db::SimpleTag {
  using type = std::string;
  static std::string name() noexcept { return "DataFileName"; }
  using option_tags =
      tmpl::list<OptionTags::DataFileName<ImporterOptionsGroup>>;
  static type create_from_options(const type& data_file_name) noexcept {
    return data_file_name;
  }
};

/*!
 * \brief The subgroup within the data file to read volume data from.
 *
 * This subgroup should conform to the `h5::VolumeData` format.
 */
template <typename ImporterOptionsGroup>
struct VolumeDataSubgroup : db::SimpleTag {
  using type = std::string;
  static std::string name() noexcept { return "VolumeDataSubgroup"; }
  using option_tags =
      tmpl::list<OptionTags::VolumeDataSubgroup<ImporterOptionsGroup>>;
  static type create_from_options(const type& volume_data_subgroup) noexcept {
    return volume_data_subgroup;
  }
};

/*!
 * \brief The observation value at which to read volume data from the file.
 */
template <typename ImporterOptionsGroup>
struct ObservationValue : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "ObservationValue"; }
  using option_tags =
      tmpl::list<OptionTags::ObservationValue<ImporterOptionsGroup>>;
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
  static std::string name() noexcept { return "RegisteredElements"; }
  using type = std::unordered_map<observers::ArrayComponentId, std::string>;
};

}  // namespace Tags

}  // namespace importer
