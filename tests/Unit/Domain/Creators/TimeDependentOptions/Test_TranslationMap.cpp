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
#include "Helpers/Domain/Creators/TimeDependent/TestHelpers.hpp"
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

namespace domain::creators::time_dependent_options {
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
    INFO("None");
    const auto translation_map_options =
        TestHelpers::test_option_tag<TranslationMapOptions<Dim>>("None");

    CHECK(not translation_map_options.has_value());
  }
  {
    INFO("Hardcoded options");
    const auto translation_map_options = TestHelpers::test_option_tag<
        domain::creators::time_dependent_options::TranslationMapOptions<Dim>>(
        "InitialValues: [" + make_array_str<Dim>(1.0) + "," +
        make_array_str<Dim>(2.0) + "," + make_array_str<Dim>(3.0) + "]");

    REQUIRE(translation_map_options.has_value());
    CHECK(std::holds_alternative<TranslationMapOptions<Dim>>(
        translation_map_options.value()));
    const std::array initial_values{DataVector{Dim, 1.0}, DataVector{Dim, 2.0},
                                    DataVector{Dim, 3.0}};
    CHECK(std::get<TranslationMapOptions<Dim>>(translation_map_options.value())
              .initial_values == initial_values);

    const auto translation_ptr =
        get_translation(translation_map_options.value(), 0.1, 65.8);

    const auto* translation =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            translation_ptr.get());

    CHECK(translation != nullptr);

    CHECK(translation->time_bounds() == std::array{0.1, 65.8});
    CHECK(translation->func_and_2_derivs(0.1) == initial_values);
  }
  {
    INFO("FromVolumeFile");
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

    TestHelpers::domain::creators::write_volume_data(filename, subfile_name,
                                                     0.1, functions_of_time);

    const auto translation_map_options = TestHelpers::test_option_tag<
        domain::creators::time_dependent_options::TranslationMapOptions<Dim>>(
        "H5Filename: " + filename + "\nSubfileName: " + subfile_name);

    REQUIRE(translation_map_options.has_value());
    CHECK(std::holds_alternative<FromVolumeFile>(
        translation_map_options.value()));
    const std::array initial_values{DataVector{Dim, 1.0}, DataVector{Dim, 2.0},
                                    DataVector{Dim, 3.0}};

    const auto translation_ptr =
        get_translation(translation_map_options.value(), 0.1, 65.8);

    const auto* translation =
        dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
            translation_ptr.get());

    CHECK(translation != nullptr);

    CHECK(translation->time_bounds() == std::array{0.1, 65.8});
    CHECK_ITERABLE_APPROX(
        translation->func_and_2_derivs(0.3),
        functions_of_time.at("Translation")->func_and_2_derivs(0.3));

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
}  // namespace domain::creators::time_dependent_options
