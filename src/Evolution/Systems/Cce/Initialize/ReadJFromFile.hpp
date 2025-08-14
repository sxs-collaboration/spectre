// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace InitializeJ {

/*!
 * \brief Load initial data from an H5 file.
 *
 * \details The data format for the H% file follows the volume H5 file
 * convention for Cce
 */
struct ReadJFromFile : InitializeJ<false> {
  struct H5Filename {
    using type = std::string;
    static constexpr Options::String help = {
        "A filename from which to retrieve a set of modes for each radial "
        "collocation point of J"};
  };
  struct SubfileNameJ {
    using type = std::string;
    static constexpr Options::String help = {
        "The subfile name inside the H5 file, e.g. 'InitialJ.dat' where data "
        "is storted"};
  };
  struct SubfileNameCoord {
    using type = std::string;
    static constexpr Options::String help = {
        "The subfile name inside the H5 file, e.g. "
        "'CartesianCauchyCoordinates.dat' where data "
        "is storted"};
  };
  struct StartTime {
    using type = double;
    static constexpr Options::String help = {
        "Start time at which read the input volume h5 file"};
  };

  using options =
      tmpl::list<H5Filename, SubfileNameJ, SubfileNameCoord, StartTime>;
  static constexpr Options::String help = {
      "Generate CCE initial data based on h5 file"};

  WRAPPED_PUPable_decl_template(ReadJFromFile);  // NOLINT
  explicit ReadJFromFile(CkMigrateMessage* msg);

  ReadJFromFile() = default;
  ReadJFromFile(std::string input_filename, std::string input_subfile_name_j,
                std::string input_subfile_name_coord, double start_time);

  std::unique_ptr<InitializeJ> get_clone() const override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta, size_t l_max,
      size_t number_of_radial_points,
      gsl::not_null<Parallel::NodeLock*> hdf5_lock) const override;

  void pup(PUP::er& p) override;

 private:
  std::string input_filename_;
  std::string input_subfile_name_j_;
  std::string input_subfile_name_coord_;
  double start_time_;
};

}  // namespace InitializeJ
}  // namespace Cce
