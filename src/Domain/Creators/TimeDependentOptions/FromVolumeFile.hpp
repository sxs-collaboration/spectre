// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
/*!
 * \brief Structs meant to be used as template parameters for the
 * `domain::creators::time_dependent_options::FromVolumeFile` classes.
 */
namespace names {
struct Translation {};
struct Rotation {};
struct Expansion {};
template <domain::ObjectLabel Object>
struct ShapeSize {};
}  // namespace names

namespace detail {
struct FromVolumeFileBase {
  struct H5Filename {
    using type = std::string;
    static constexpr Options::String help{
        "Name of H5 file to read functions of time from."};
  };

  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help{
        "Subfile that holds the volume data. Must be an h5::VolumeData "
        "subfile."};
  };

  struct Time {
    using type = double;
    static constexpr Options::String help =
        "Time in the H5File to get the coefficients at. Will likely be the "
        "same as the initial time";
  };

  using options = tmpl::list<H5Filename, SubfileName, Time>;
  static constexpr Options::String help =
      "Read function of time coefficients from a volume subfile of an H5 file.";

  FromVolumeFileBase() = default;
};
}  // namespace detail

/// @{
/*!
 * \brief Read in FunctionOfTime coefficients from an H5 file and volume
 * subfile.
 *
 * \details To use, template the class on one of the structs in
 * `domain::creators::time_dependent_options::names`. The general struct will
 * have one member, `values` that will hold the function of time and its first
 * two derivatives.
 *
 * There are specializations for
 *
 * - `domain::creators::time_dependent_options::names::Rotation` because of
 * quaternions
 * - `domain::creators::time_dependent_options::name::Expansion` because it also
 * has outer boundary values (a second function of time)
 * - `domain::creators::time_dependent_options::names::ShapeSize` because it
 * handles both the Shape and Size function of time.
 */
template <typename FoTName>
struct FromVolumeFile : public detail::FromVolumeFileBase {
  FromVolumeFile() = default;
  FromVolumeFile(const std::string& h5_filename,
                 const std::string& subfile_name, double time,
                 const Options::Context& context = {});

  std::array<DataVector, 3> values{};
};

template <>
struct FromVolumeFile<names::Expansion> : public detail::FromVolumeFileBase {
  FromVolumeFile() = default;
  FromVolumeFile(const std::string& h5_filename,
                 const std::string& subfile_name, double time,
                 const Options::Context& context = {});

  std::array<DataVector, 3> expansion_values{};
  std::array<DataVector, 3> expansion_values_outer_boundary{};
};

template <>
struct FromVolumeFile<names::Rotation> : public detail::FromVolumeFileBase {
  FromVolumeFile() = default;
  FromVolumeFile(const std::string& h5_filename,
                 const std::string& subfile_name, double time,
                 const Options::Context& context = {});

  std::array<DataVector, 3> quaternions{};
  std::array<DataVector, 4> angle_values{};
};

template <ObjectLabel Object>
struct FromVolumeFile<names::ShapeSize<Object>>
    : public detail::FromVolumeFileBase {
  FromVolumeFile() = default;
  FromVolumeFile(const std::string& h5_filename,
                 const std::string& subfile_name, double time,
                 const Options::Context& context = {});

  std::array<DataVector, 3> shape_values{};
  std::array<DataVector, 4> size_values{};
};
/// @}
}  // namespace domain::creators::time_dependent_options
