// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/Initialize/ReadJFromFile.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::InitializeJ {

ReadJFromFile::ReadJFromFile(CkMigrateMessage* msg) : InitializeJ<false>(msg) {}

ReadJFromFile::ReadJFromFile(std::string input_filename,
                             std::string input_subfile_name)
    : input_filename_{std::move(input_filename)},
      input_subfile_name_{std::move(input_subfile_name)} {}

std::unique_ptr<InitializeJ<false>> ReadJFromFile::get_clone() const {
  return std::make_unique<ReadJFromFile>(*this);
}

void ReadJFromFile::operator()(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
    const gsl::not_null<
        tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
        angular_cauchy_coordinates,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*boundary_j*/,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& /*boundary_dr_j*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*r*/,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& /*beta*/,
    const size_t l_max, const size_t number_of_radial_points,
    const gsl::not_null<Parallel::NodeLock*> /*hdf5_lock*/
) const {
  // Reading volume data from "input_file_name_" h5
  // file path at the first observation id (initial time).

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file{input_filename_};
  auto& dat_file = cce_data_file.get<h5::VolumeData>(input_subfile_name_);
  std::vector<size_t> obs_ids = dat_file.list_observation_ids();
  if (obs_ids.size() == 0) {
    ERROR("The observation IDs list is empty");
  }
  const auto& j_tensor_component =
      dat_file.get_tensor_component(obs_ids[0], "J");

  if (not std::holds_alternative<DataVector>(j_tensor_component.data)) {
    ERROR("CCE initial J must be a DataVector");
  }

  // Construct a DataVector containing the real and imaginary parts of the
  // Goldberg modes of J for each radial collocation points

  const auto& modal_j_goldberg_real =
      std::get<DataVector>(j_tensor_component.data);
  cce_data_file.close_current_object();
  const size_t l_max_plus_one_squared = square(l_max + 1);
  if (modal_j_goldberg_real.size() !=
      2 * number_of_radial_points * l_max_plus_one_squared) {
    ERROR(
        std::string("Mismatch between l_max or number of radial points between "
                    "J read from h5 and the input settings. j_data_size: "));
  }

  // Recasting the DataVector (real) into a ComplexModalVector (complex).
  // Both these sets of modes are in Goldberg convention

  SpinWeighted<ComplexModalVector, 2> modal_j_goldberg{number_of_radial_points *
                                                       l_max_plus_one_squared};
  for (size_t i = 0; i < modal_j_goldberg.size(); i++) {
    modal_j_goldberg.data()[i] = std::complex<double>(
        modal_j_goldberg_real[2 * i], modal_j_goldberg_real[(2 * i) + 1]);
  }

  // Definition of a new ComplexModalVector that will contain the modal values
  //  of J in libsharp convention (used for actual computations during a CCE
  //  run)

  const size_t number_of_libsharp_modes =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  SpinWeighted<ComplexModalVector, 2> modal_j_libsharp{
      number_of_radial_points * number_of_libsharp_modes};

  // Here we convert the goldberg modal values for J obtained by the h5 to
  // libsharp modal values, this is done for each radial collocation points
  // using non-owning angular views

  for (size_t i = 0; i < number_of_radial_points; i++) {
    SpinWeighted<ComplexModalVector, 2> angular_view_j_goldberg;
    auto& complex_data_vector_view_goldberg = angular_view_j_goldberg.data();
    complex_data_vector_view_goldberg.set_data_ref(
        modal_j_goldberg.data().data() + (l_max_plus_one_squared * i),
        l_max_plus_one_squared);
    SpinWeighted<ComplexModalVector, 2> angular_view_j_libsharp;
    auto& complex_data_vector_view_libsharp = angular_view_j_libsharp.data();
    complex_data_vector_view_libsharp.set_data_ref(
        modal_j_libsharp.data().data() + (number_of_libsharp_modes * i),
        number_of_libsharp_modes);
    angular_view_j_libsharp = Spectral::Swsh::goldberg_to_libsharp_modes(
        angular_view_j_goldberg, l_max);
  }

  // The initial J constructed by any initial data scheme has to be provided in
  // nodal angular values, so the libsharp modal values of J are converted into
  // angular grid points values

  get(*j) = Spectral::Swsh::inverse_swsh_transform(
      l_max, number_of_radial_points, modal_j_libsharp);
  Parallel::printf("%s\n", get(*j).data()[8]);

  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      cartesian_cauchy_coordinates, angular_cauchy_coordinates, l_max);
}

void ReadJFromFile::pup(PUP::er& p) {
  p | input_filename_;
  p | input_subfile_name_;
}

PUP::able::PUP_ID ReadJFromFile::my_PUP_ID = 0;

}  // namespace Cce::InitializeJ
