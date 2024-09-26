// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
template <size_t Dim>
std::string make_array_str(const double value) {
  std::stringstream ss{};
  ss << "[" << value;
  if constexpr (Dim > 1) {
    ss << ", " << value;
    if constexpr (Dim > 2) {
      ss << ", " << value;
    }
  }
  ss << "]";

  return ss.str();
}

template <size_t Dim>
void test_translation_map_options() {
  {
    const auto translation_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::TranslationMapOptions<Dim>>(
        "InitialValues: [" + make_array_str<Dim>(1.0) + "," +
        make_array_str<Dim>(2.0) + "," + make_array_str<Dim>(3.0) + "]");
    CHECK(translation_map_options.name() == "TranslationMap");
    CHECK(translation_map_options.initial_values ==
          std::array{DataVector{Dim, 1.0}, DataVector{Dim, 2.0},
                     DataVector{Dim, 3.0}});
  }
  {
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};
    functions_of_time["Translation"] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            0.0,
            std::array{DataVector{Dim, 1.0}, DataVector{Dim, 2.0},
                       DataVector{Dim, 3.0}},
            100.0);
    const std::string filename{"NewFinalActualFinal2Final.h5"};
    const std::string subfile_name{"VolumeData"};
    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }

    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{filename};
      auto& vol_file = h5_file.insert<h5::VolumeData>(subfile_name);

      // We don't care about the volume data here, just the functions of time
      vol_file.write_volume_data(
          0, 0.0,
          {ElementVolumeData{
              "blah",
              {TensorComponent{"RandomTensor", DataVector{3, 0.0}}},
              {3},
              {Spectral::Basis::Legendre},
              {Spectral::Quadrature::GaussLobatto}}},
          std::nullopt, serialize(functions_of_time));
    }

    const auto translation_map_options = TestHelpers::test_creation<
        domain::creators::time_dependent_options::TranslationMapOptions<Dim>>(
        "InitialValues:\n"
        "  H5Filename: " +
        filename + "\n  SubfileName: " + subfile_name + "\n  Time: 0.0");
    CHECK(translation_map_options.name() == "TranslationMap");
    CHECK(translation_map_options.initial_values ==
          std::array{DataVector{Dim, 1.0}, DataVector{Dim, 2.0},
                     DataVector{Dim, 3.0}});

    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.TranslationMap",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  test_translation_map_options<1>();
  test_translation_map_options<2>();
  test_translation_map_options<3>();
}
