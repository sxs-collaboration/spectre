// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "IO/Importers/ObservationSelector.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TaggedTuple.hpp"

/// Items related to loading data from files
namespace importers {

/// The input file options associated with the data importer
namespace OptionTags {

/*!
 * \brief The file to read data from.
 */
struct FileGlob {
  using type = std::string;
  static constexpr Options::String help = "Path to the data file";
};

/*!
 * \brief The subgroup within the file to read data from.
 *
 * This subgroup should conform to the `h5::VolumeData` format.
 */
struct Subgroup {
  using type = std::string;
  static constexpr Options::String help =
      "The subgroup within the file, excluding extensions";
};

/*!
 * \brief The observation value at which to read data from the file.
 */
struct ObservationValue {
  using type = std::variant<double, ObservationSelector>;
  static constexpr Options::String help =
      "The observation value at which to read data";
};

struct ObservationValueEpsilon {
  using type = Options::Auto<double>;
  static constexpr Options::String help =
      "Look for observations in the data within this epsilon of the "
      "'ObservationValue'. Set to 'Auto' to use default of 1e-12. This option "
      "is ignored if the 'ObservationValue' is a selector like 'First' or "
      "'Last'.";
};

/*!
 * \brief Toggle interpolation of numeric data to the target domain
 */
struct ElementsMatch {
  using type = bool;
  static constexpr Options::String help =
      "Indicate that the elements of the source and target domain are the "
      "same, meaning that the domains are the same and their h-refinement is "
      "the same. In this case, data can be transferred between the source and "
      "target elements one-to-one, and interpolations only happen if the "
      "elements have different meshes (p-refinement). "
      "For example, you can enable this option if you have generated data "
      "on the target points, or if you have already interpolated your data, "
      "or if you import data from a simulation that differs only by "
      "p-refinement. "
      "When this option is enabled, datasets "
      "'InertialCoordinates(_x,_y,_z)' must exist in the files. They are used "
      "to verify that the target points indeed match the source data.";
};
}  // namespace OptionTags

/// Options that specify the volume data to load. See the option tags for
/// details.
struct ImporterOptions
    : tuples::TaggedTuple<OptionTags::FileGlob, OptionTags::Subgroup,
                          OptionTags::ObservationValue,
                          OptionTags::ObservationValueEpsilon,
                          OptionTags::ElementsMatch> {
  using options = tags_list;
  static constexpr Options::String help = "The volume data to load.";
  using TaggedTuple::TaggedTuple;
};

/// The \ref DataBoxGroup tags associated with the data importer
namespace Tags {

/// Options that specify the volume data to load. See the option tags for
/// details.
template <typename OptionsGroup>
struct ImporterOptions : db::SimpleTag {
  static std::string name() { return "VolumeData"; }
  using type = importers::ImporterOptions;
  static constexpr Options::String help = importers::ImporterOptions::help;
  using group = OptionsGroup;
  using option_tags = tmpl::list<ImporterOptions>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(type value) { return value; }
};

/*!
 * \brief The elements that will receive data from the importer.
 *
 * \details Identifiers for elements from multiple parallel components can be
 * stored. Each element is identified by an `Parallel::ArrayComponentId` and
 * also needs to provide the inertial coordinates of its grid points. The
 * imported data will be interpolated to these grid points.
 */
template <size_t Dim>
struct RegisteredElements : db::SimpleTag {
  using type = std::unordered_map<
      Parallel::ArrayComponentId,
      std::pair<tnsr::I<DataVector, Dim, Frame::Inertial>, ::Mesh<Dim>>>;
};

/// Indicates which volume data files have already been read.
struct ElementDataAlreadyRead : db::SimpleTag {
  using type = std::unordered_set<size_t>;
};

/*!
 * \brief Inbox tag that carries the data read from a volume data file.
 *
 * Since we read a volume data file only once, this tag's map will only ever
 * hold data at the index (i.e. the temporal ID) with value `0`.
 */
template <typename FieldTagsList>
struct VolumeData : Parallel::InboxInserters::Value<VolumeData<FieldTagsList>> {
  using temporal_id = size_t;
  using type =
      std::map<temporal_id, tuples::tagged_tuple_from_typelist<FieldTagsList>>;

  static std::string output_inbox(const type& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');

    ss << std::scientific << std::setprecision(16);
    ss << pad << "VolumeDataInbox:\n";
    // We don't really care about the variables, just the times
    for (const auto& [index, variables] : inbox) {
      (void)variables;
      ss << pad << " Index: " << index << "\n";
    }

    return ss.str();
  }
};
}  // namespace Tags

}  // namespace importers
