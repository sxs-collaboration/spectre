// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Exporter/InterpolateToPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/*!
 * \brief Load numeric data from volume data files
 *
 * This class loads all requested tags from volume data files using
 * `spectre::Exporter::interpolate_to_points`. This is an easy and useful
 * alternative to `importers::ElementDataReader` to load numeric data with the
 * following advantages:
 *
 * - No need to work with parallel components, actions, or phases to load
 *   numeric data. Also, no need to parse numeric data options separately from
 *   analytic data options. Just include this class in the list of analytic data
 *   classes for the executable.
 * - The data is interpolated and returned in serial, just like analytic data.
 * - The data can be re-interpolated to any set of points on request, which is
 *   an easy way to handle AMR or other domain changes.
 *
 * However, it also comes with the following caveats:
 *
 * - The volume data files must have datasets with the same names as the tags
 *   being loaded. For example, if the tag being loaded is `gr::Tags::Shift`,
 *   then the volume data files must have datasets named `Shift_x`, `Shift_y`,
 *   and `Shift_z`. If you need processing of the data before it can be used,
 *   e.g. to load some datasets and compute other derived quantities from them,
 *   then consider writing a custom class or use `importers::ElementDataReader`.
 * - Reading in data on the same grid that it was written on may not give you
 *   the same data back, at least not on Gauss-Lobatto grids. This is because
 *   grid points on element boundaries are disambiguated by
 *   `block_logical_coordinates` and `element_logical_coordinates`, so a
 *   boundary point may be written by one element but read back in from its
 *   neighbor. To avoid this, consider using `importers::ElementDataReader`
 *   which has functionality to read in data on the same grid that it was
 *   written on. This is not possible with this class because it is not aware of
 *   the element structure (it operates in a pointwise manner).
 * - Large datasets may not fit in memory. Each element will open the H5 files
 *   and interpolate data from them, so this may not fit in memory if the
 *   datasets are large and/or many elements are doing this at the same time. To
 *   avoid this, consider using `importers::ElementDataReader` which reads one
 *   H5 file at a time on the node level.
 */
class NumericData {
 public:
  struct FileGlob {
    static constexpr Options::String help =
        "Path or glob pattern to the data file";
    using type = std::string;
  };
  struct Subgroup {
    static constexpr Options::String help = {
        "The subgroup within the file, excluding extensions"};
    using type = std::string;
  };
  struct ObservationStep {
    static constexpr Options::String help =
        "The observation step at which to read data";
    using type = int;
  };
  struct ExtrapolateIntoExcisions {
    static constexpr Options::String help = {
        "Whether to extrapolate data into excised regions"};
    using type = bool;
  };
  using options =
      tmpl::list<FileGlob, Subgroup, ObservationStep, ExtrapolateIntoExcisions>;
  static constexpr Options::String help =
      "Numeric data loaded from volume data files";

  NumericData() = default;
  NumericData(const NumericData&) = default;
  NumericData& operator=(const NumericData&) = default;
  NumericData(NumericData&&) = default;
  NumericData& operator=(NumericData&&) = default;
  ~NumericData() = default;

  NumericData(std::string file_glob, std::string subgroup, int observation_step,
              bool extrapolate_into_excisions);

  const std::string& file_glob() const { return file_glob_; }

  const std::string& subgroup() const { return subgroup_; }

  int observation_step() const { return observation_step_; }

  bool extrapolate_into_excisions() const {
    return extrapolate_into_excisions_;
  }

  template <typename DataType, size_t Dim, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return spectre::Exporter::interpolate_to_points<
        tmpl::list<RequestedTags...>>(
        file_glob_, subgroup_,
        spectre::Exporter::ObservationStep{observation_step_}, x,
        extrapolate_into_excisions_);
  }

  template <size_t Dim, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x, const Mesh<Dim>& /*mesh*/,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p);

 protected:
  std::string file_glob_{};
  std::string subgroup_{};
  int observation_step_{};
  bool extrapolate_into_excisions_{};
};

bool operator==(const NumericData& lhs, const NumericData& rhs);
bool operator!=(const NumericData& lhs, const NumericData& rhs);

namespace elliptic::analytic_data {

/*!
 * \brief Load numeric data from volume data files
 *
 * \see ::NumericData
 */
class NumericData : public elliptic::analytic_data::Background,
                    public elliptic::analytic_data::InitialGuess,
                    public ::NumericData {
 public:
  using ::NumericData::NumericData;

  explicit NumericData(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumericData);

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;
};

bool operator==(const NumericData& lhs, const NumericData& rhs);
bool operator!=(const NumericData& lhs, const NumericData& rhs);

}  // namespace elliptic::analytic_data

namespace evolution::initial_data {

/*!
 * \brief Load numeric data from volume data files
 *
 * \see ::NumericData
 */
class NumericData : public evolution::initial_data::InitialData,
                    public ::NumericData {
 public:
  using ::NumericData::NumericData;

  explicit NumericData(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumericData);

  std::unique_ptr<evolution::initial_data::InitialData> get_clone()
      const override;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;
};

bool operator==(const NumericData& lhs, const NumericData& rhs);
bool operator!=(const NumericData& lhs, const NumericData& rhs);

}  // namespace evolution::initial_data
