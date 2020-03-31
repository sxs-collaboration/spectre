// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
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

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups options for reading in FunctionOfTime data from SpEC
 */
struct SpecFuncOfTimeReader {
  static constexpr OptionString help{
      "Options for importing FunctionOfTimes from SpEC"};
};

/*!
 * \brief Path to an H5 file containing SpEC FunctionOfTime data
 */
struct FunctionOfTimeFile {
  using type = std::string;
  static constexpr OptionString help{
      "Path to an H5 file containing SpEC FunctionOfTime data"};
  using group = SpecFuncOfTimeReader;
};

/*!
 * \brief Pairs of strings mapping SpEC FunctionOfTime names to SpECTRE names
 */
struct FunctionOfTimeNameMap {
  using type = std::map<std::string, std::string>;
  static constexpr OptionString help{
      "String pairs mapping spec names to spectre names"};
  using group = SpecFuncOfTimeReader;
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

/*!
 * \brief Path to an H5 file containing SpEC `FunctionOfTime` data to read.
 */
struct FunctionOfTimeFile : db::SimpleTag {
  static std::string name() noexcept { return "FunctionOfTimeFile"; }
  using type = std::string;
  using option_tags = tmpl::list<::importers::OptionTags::FunctionOfTimeFile>;
  static constexpr bool pass_metavariables = false;
  template <typename Metavariables>
  static std::string create_from_options(
      const std::string& function_of_time_file) noexcept {
    return function_of_time_file;
  }
};

/*!
 * \brief Pairs of strings mapping SpEC -> SpECTRE FunctionOfTime names
 *
 * \details The first string in each pair is the name of a Dat file inside
 * an H5 file that contains SpEC FunctionOfTime data.
 * The second string in each pair is the SpECTRE name of the FunctionOfTime,
 * which will be the key used to index the `FunctionOfTime` in a
 * `std::unordered_map` after reading it.
 */
struct FunctionOfTimeNameMap : db::SimpleTag {
  static std::string name() noexcept { return "FunctionOfTimeNameMap"; }
  using type = std::map<std::string, std::string>;
  using option_tags =
      tmpl::list<::importers::OptionTags::FunctionOfTimeNameMap>;
  static constexpr bool pass_metavariables = false;
  template <typename Metavariables>
  static std::map<std::string, std::string> create_from_options(
      const std::map<std::string, std::string>& dataset_names) noexcept {
    return dataset_names;
  }
};

}  // namespace Tags

}  // namespace importers
