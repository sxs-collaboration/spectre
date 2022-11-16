// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "IO/Importers/ObservationSelector.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Options/Options.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TaggedTuple.hpp"

/// Items related to loading data from files
namespace importers {

/// The input file options associated with the data importer
namespace OptionTags {

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups the data importer configurations in the input file
 */
struct Group {
  static std::string name() { return "Importers"; }
  static constexpr Options::String help = "Options for loading data files";
};

/*!
 * \brief The file to read data from.
 */
template <typename ImporterOptionsGroup>
struct FileGlob {
  static_assert(
      std::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'Importers' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = std::string;
  static constexpr Options::String help = "Path to the data file";
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
      std::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'Importers' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = std::string;
  static constexpr Options::String help =
      "The subgroup within the file, excluding extensions";
  using group = ImporterOptionsGroup;
};

/*!
 * \brief The observation value at which to read data from the file.
 */
template <typename ImporterOptionsGroup>
struct ObservationValue {
  static_assert(
      std::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'Importers' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = std::variant<double, ObservationSelector>;
  static constexpr Options::String help =
      "The observation value at which to read data";
  using group = ImporterOptionsGroup;
};

/*!
 * \brief Toggle interpolation of numeric data to the target domain
 */
template <typename ImporterOptionsGroup>
struct EnableInterpolation {
  static std::string name() { return "Interpolate"; }
  static_assert(
      std::is_same_v<typename ImporterOptionsGroup::group, Group>,
      "The importer options should be placed in the 'Importers' option "
      "group. Add a type alias `using group = importers::OptionTags::Group`.");
  using type = bool;
  static constexpr Options::String help =
      "Enable to interpolate the volume data to the target domain. Disable to "
      "load volume data directly into elements with the same name. "
      "For example, you can disable interpolation if you have generated data "
      "on the target points, or if you have already interpolated your data. "
      "When interpolation is disabled, datasets "
      "'InertialCoordinates(_x,_y,_z)' must exist in the files. They are used "
      "to verify that the target points indeed match the source data.";
  using group = ImporterOptionsGroup;
};
}  // namespace OptionTags

/// The \ref DataBoxGroup tags associated with the data importer
namespace Tags {

/*!
 * \brief The file to read data from.
 */
template <typename ImporterOptionsGroup>
struct FileGlob : db::SimpleTag {
  static std::string name() {
    return "FileGlob(" + pretty_type::name<ImporterOptionsGroup>() + ")";
  }
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::FileGlob<ImporterOptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& file_name) { return file_name; }
};

/*!
 * \brief The subgroup within the file to read data from.
 *
 * This subgroup should conform to the `h5::VolumeData` format.
 */
template <typename ImporterOptionsGroup>
struct Subgroup : db::SimpleTag {
  static std::string name() {
    return "Subgroup(" + pretty_type::name<ImporterOptionsGroup>() + ")";
  }
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::Subgroup<ImporterOptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& subgroup) { return subgroup; }
};

/*!
 * \brief The observation value at which to read data from the file.
 */
template <typename ImporterOptionsGroup>
struct ObservationValue : db::SimpleTag {
  static std::string name() {
    return "ObservationValue(" + pretty_type::name<ImporterOptionsGroup>() +
           ")";
  }
  using type = std::variant<double, ObservationSelector>;
  using option_tags =
      tmpl::list<OptionTags::ObservationValue<ImporterOptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& observation_value) {
    return observation_value;
  }
};

/*!
 * \brief Toggle interpolation of numeric data to the target domain
 */
template <typename ImporterOptionsGroup>
struct EnableInterpolation : db::SimpleTag {
  static std::string name() {
    return "EnableInterpolation(" + pretty_type::name<ImporterOptionsGroup>() +
           ")";
  }
  using type = bool;
  using option_tags =
      tmpl::list<OptionTags::EnableInterpolation<ImporterOptionsGroup>>;

  static constexpr bool pass_metavariables = false;
  static bool create_from_options(const bool enable_interpolation) {
    return enable_interpolation;
  }
};

/*!
 * \brief The elements that will receive data from the importer.
 *
 * \details Identifiers for elements from multiple parallel components can be
 * stored. Each element is identified by an `observers::ArrayComponentId` and
 * also needs to provide the inertial coordinates of its grid points. The
 * imported data will be interpolated to these grid points.
 */
template <size_t Dim>
struct RegisteredElements : db::SimpleTag {
  using type = std::unordered_map<observers::ArrayComponentId,
                                  tnsr::I<DataVector, Dim, Frame::Inertial>>;
};

/// Indicates which volume data files have already been read.
struct ElementDataAlreadyRead : db::SimpleTag {
  using type = std::unordered_set<std::string>;
};

/*!
 * \brief Inbox tag that carries the data read from a volume data file.
 *
 * Since we read a volume data file only once, this tag's map will only ever
 * hold data at the index (i.e. the temporal ID) with value `0`.
 */
template <typename ImporterOptionsGroup, typename FieldTagsList>
struct VolumeData : Parallel::InboxInserters::Value<
                        VolumeData<ImporterOptionsGroup, FieldTagsList>> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id, tuples::tagged_tuple_from_typelist<FieldTagsList>>;
};
}  // namespace Tags

}  // namespace importers
