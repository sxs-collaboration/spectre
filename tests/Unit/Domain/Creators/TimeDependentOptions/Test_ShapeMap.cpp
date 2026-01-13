// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TimeDependent/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators::time_dependent_options {
namespace {
void test_kerr_schild_boyer_lindquist() {
  const auto kerr_schild_boyer_lindquist = TestHelpers::test_creation<
      domain::creators::time_dependent_options::KerrSchildFromBoyerLindquist>(
      "Mass: 1.7\n"
      "Spin: [0.45, 0.12, 0.34]");
  CHECK(kerr_schild_boyer_lindquist.mass == 1.7);
  CHECK(kerr_schild_boyer_lindquist.spin == std::array{0.45, 0.12, 0.34});
}

void test_ylms_from_file() {
  const auto ylms_from_file = TestHelpers::test_creation<
      domain::creators::time_dependent_options::YlmsFromFile>(
      "H5Filename: TotalEclipseOfTheHeart.h5\n"
      "SubfileNames:\n"
      "  - Ylm_coefs\n"
      "  - dt_Ylm_coefs\n"
      "MatchTime: 1.7\n"
      "MatchTimeEpsilon: 1.0e-14\n"
      "CheckFrame: True\n"
      "SetL1CoefsToZero: False");
  CHECK(ylms_from_file.h5_filename == "TotalEclipseOfTheHeart.h5");
  CHECK(ylms_from_file.subfile_names ==
        std::vector<std::string>{"Ylm_coefs", "dt_Ylm_coefs"});
  CHECK(ylms_from_file.match_time == 1.7);
  CHECK(ylms_from_file.match_time_epsilon.has_value());
  CHECK(ylms_from_file.match_time_epsilon.value() == 1.0e-14);
  CHECK_FALSE(ylms_from_file.set_l1_coefs_to_zero);
}

template <typename Generator>
void test_funcs(const gsl::not_null<Generator*> generator) {
  const double inner_radius = 0.5;
  const size_t l_max = 8;
  {
    INFO("None");
    const auto shape_map_options = TestHelpers::test_option_tag<
        ShapeMapOptions<false, domain::ObjectLabel::None>>("None");

    CHECK(not shape_map_options.has_value());
  }
  {
    INFO("Spherical");
    const auto shape_map_options = TestHelpers::test_option_tag<
        ShapeMapOptions<false, domain::ObjectLabel::None>>(
        "LMax: 8\n"
        "CoefficientTruncationLimit: 0.0\n"
        "InitialValues: Spherical\n"
        "SizeInitialValues: Auto\n");

    REQUIRE(shape_map_options.has_value());
    REQUIRE(std::holds_alternative<
            ShapeMapOptions<false, domain::ObjectLabel::None>>(
        shape_map_options.value()));

    const FunctionsOfTimeMap shape_and_size = get_shape_and_size(
        shape_map_options.value(), 0.1, 0.8, 0.9, inner_radius);

    CHECK(shape_and_size.at("Shape")->time_bounds() == std::array{0.1, 0.8});
    CHECK(shape_and_size.at("Size")->time_bounds() == std::array{0.1, 0.9});

    CHECK(shape_and_size.at("Shape")->func_and_2_derivs(0.1) ==
          make_array<3>(
              DataVector{ylm::Spherepack::spectral_size(l_max, l_max), 0.0}));
    CHECK(shape_and_size.at("Size")->func_and_2_derivs(0.1) ==
          make_array<3>(DataVector{0.0}));
  }
  // We choose a Schwarzschild BH so all coefs are zero and it's easy to check
  {
    INFO("KerrSchild");
    const auto shape_map_options = TestHelpers::test_option_tag<
        domain::creators::time_dependent_options::ShapeMapOptions<
            false, domain::ObjectLabel::None>>(
        "LMax: 8\n"
        "CoefficientTruncationLimit: 0.0\n"
        "InitialValues:\n"
        "  Mass: 1.0\n"
        "  Spin: [0.0, 0.0, 0.0]\n"
        "SizeInitialValues: [0.5, 1.0, 2.4]");

    REQUIRE(shape_map_options.has_value());
    REQUIRE(std::holds_alternative<
            ShapeMapOptions<false, domain::ObjectLabel::None>>(
        shape_map_options.value()));

    const FunctionsOfTimeMap shape_and_size = get_shape_and_size(
        shape_map_options.value(), 0.1, 0.8, 0.9, inner_radius);

    CHECK(shape_and_size.at("Shape")->time_bounds() == std::array{0.1, 0.8});
    CHECK(shape_and_size.at("Size")->time_bounds() == std::array{0.1, 0.9});

    CHECK(shape_and_size.at("Shape")->func_and_2_derivs(0.1) ==
          make_array<3>(
              DataVector{ylm::Spherepack::spectral_size(l_max, l_max), 0.0}));
    CHECK(shape_and_size.at("Size")->func_and_2_derivs(0.1) ==
          std::array{DataVector{0.5}, DataVector{1.0}, DataVector{2.4}});
  }
  {
    const std::string test_filename{"TotalEclipseOfTheHeart.h5"};
    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }
    const std::vector<std::string> subfile_names{"Ylm_coefs", "dt_Ylm_coefs"};
    const std::string volume_subfile_name{"VolumeData"};
    const double time = 1.7;
    // Purposefully larger than the LMax in the options so that the
    // Strahlkorpers will be restricted
    const size_t file_l_max = 10;
    std::uniform_real_distribution<double> distribution(0.1, 2.0);
    using Frame = ::Frame::Inertial;
    std::array<ylm::Strahlkorper<Frame>, 3> strahlkorpers{};
    std::vector<std::string> legend{};
    std::vector<double> data{};

    // Scoped to close h5 file
    {
      h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename, true);
      for (size_t i = 0; i < 3; i++) {
        legend.clear();
        data.clear();
        const auto radius = make_with_random_values<DataVector>(
            generator, distribution,
            DataVector(ylm::Spherepack::physical_size(file_l_max, file_l_max),
                       std::numeric_limits<double>::signaling_NaN()));
        if (i < 2) {
          gsl::at(strahlkorpers, i) = ylm::Strahlkorper<Frame>(
              file_l_max, file_l_max, radius, std::array{0.0, 0.0, 0.0});
        } else {
          gsl::at(strahlkorpers, i) = ylm::Strahlkorper<Frame>(
              file_l_max, inner_radius, std::array{0.0, 0.0, 0.0});
          continue;
        }

        ylm::fill_ylm_legend_and_data(
            make_not_null(&legend), make_not_null(&(data)),
            gsl::at(strahlkorpers, i), time, file_l_max);

        auto& file = test_file.insert<h5::Dat>("/" + subfile_names[i], legend);
        file.append(data);
        test_file.close_current_object();
      }
    }

    // Take advantage of the funcs and write those to the volume file
    const std::string shape_name{"ShapeB"};
    const std::string size_name{"SizeB"};
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        functions_of_time{};
    {
      INFO("YlmsFromFile specify size");
      const auto shape_map_options = TestHelpers::test_option_tag<
          domain::creators::time_dependent_options::ShapeMapOptions<
              true, domain::ObjectLabel::B>>(
          "LMax: 8\n"
          "CoefficientTruncationLimit: 0.0\n"
          "InitialValues:\n"
          "  H5Filename: TotalEclipseOfTheHeart.h5\n"
          "  SubfileNames:\n"
          "    - Ylm_coefs\n"
          "    - dt_Ylm_coefs\n"
          "  MatchTime: 1.7\n"
          "  MatchTimeEpsilon: Auto\n"
          "  SetL1CoefsToZero: True\n"
          "  CheckFrame: False\n"
          "SizeInitialValues: [1.1, 2.2, 3.3]\n"
          "TransitionEndsAtCube: True");

      REQUIRE(shape_map_options.has_value());
      REQUIRE(
          std::holds_alternative<ShapeMapOptions<true, domain::ObjectLabel::B>>(
              shape_map_options.value()));

      const FunctionsOfTimeMap shape_and_size = get_shape_and_size(
          shape_map_options.value(), 0.1, 0.8, 0.9, inner_radius);

      CHECK(shape_and_size.at(shape_name)->time_bounds() ==
            std::array{0.1, 0.8});
      CHECK(shape_and_size.at(size_name)->time_bounds() ==
            std::array{0.1, 0.9});

      const auto shape_funcs =
          shape_and_size.at(shape_name)->func_and_2_derivs(0.1);
      const auto size_funcs =
          shape_and_size.at(size_name)->func_and_2_derivs(0.1);

      ylm::SpherepackIterator iter{l_max, l_max};
      ylm::SpherepackIterator file_iter{file_l_max, file_l_max};
      for (size_t i = 0; i < shape_funcs.size(); i++) {
        const size_t expected_size =
            ylm::Spherepack::spectral_size(l_max, l_max);
        // Make sure they were restricted properly
        CHECK(gsl::at(shape_funcs, i).size() == expected_size);

        // Loop pointwise so we only check the coefficients that matter
        for (size_t l = 0; l <= l_max; l++) {
          for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
            CAPTURE(l);
            CAPTURE(m);
            const double expected_value =
                l < 2 ? 0.0
                      : -1.0 * gsl::at(strahlkorpers, i)
                                   .coefficients()[file_iter.set(l, m)()];
            CHECK(gsl::at(shape_funcs, i)[iter.set(l, m)()] == expected_value);
          }
        }
      }
      CHECK(size_funcs ==
            std::array{DataVector{1.1}, DataVector{2.2}, DataVector{3.3}});

      functions_of_time[shape_name] =
          shape_and_size.at(shape_name)->get_clone();
      functions_of_time[size_name] = shape_and_size.at(size_name)->get_clone();

      // For the FromVolumeFile test later on
      TestHelpers::domain::creators::write_volume_data(
          test_filename, volume_subfile_name, 0.2, functions_of_time);
    }

    // Already checked that shape funcs are correct. Here just check that size
    // funcs were automatically set to correct values
    {
      INFO("YlmsFromFile auto size");
      const auto shape_map_options = TestHelpers::test_option_tag<
          domain::creators::time_dependent_options::ShapeMapOptions<
              true, domain::ObjectLabel::B>>(
          "LMax: 8\n"
          "CoefficientTruncationLimit: 0.0\n"
          "InitialValues:\n"
          "  H5Filename: TotalEclipseOfTheHeart.h5\n"
          "  SubfileNames:\n"
          "    - Ylm_coefs\n"
          "    - dt_Ylm_coefs\n"
          "  MatchTime: 1.7\n"
          "  MatchTimeEpsilon: Auto\n"
          "  SetL1CoefsToZero: True\n"
          "  CheckFrame: False\n"
          "SizeInitialValues: Auto\n"
          "TransitionEndsAtCube: true");

      REQUIRE(shape_map_options.has_value());
      REQUIRE(
          std::holds_alternative<ShapeMapOptions<true, domain::ObjectLabel::B>>(
              shape_map_options.value()));

      const FunctionsOfTimeMap shape_and_size = get_shape_and_size(
          shape_map_options.value(), 0.1, 0.8, 0.9, inner_radius);

      CHECK_ITERABLE_APPROX(
          shape_and_size.at(size_name)->func_and_2_derivs(0.1),
          (std::array{
              DataVector{(inner_radius - ylm::Spherepack::average(
                                             strahlkorpers[0].coefficients())) *
                         2.0 * sqrt(M_PI)},
              DataVector{(inner_radius - ylm::Spherepack::average(
                                             strahlkorpers[1].coefficients())) *
                         2.0 * sqrt(M_PI)},
              DataVector{0.0}}));
    }

    {
      INFO("FromVolumeFile");
      const auto shape_map_options = TestHelpers::test_option_tag<
          domain::creators::time_dependent_options::ShapeMapOptions<
              false, domain::ObjectLabel::B>>(
          "LMax: 6\n"
          "CoefficientTruncationLimit: 0.0\n"
          "TransitionEndsAtCube: false\n"
          "H5Filename: TotalEclipseOfTheHeart.h5\n"
          "SubfileName: VolumeData\n");

      REQUIRE(shape_map_options.has_value());
      REQUIRE(std::holds_alternative<
              FromVolumeFileShapeSize<domain::ObjectLabel::B>>(
          shape_map_options.value()));

      CHECK(std::get<FromVolumeFileShapeSize<domain::ObjectLabel::B>>(
                shape_map_options.value())
                .l_max == 6);

      const FunctionsOfTimeMap shape_and_size = get_shape_and_size(
          shape_map_options.value(), 0.2, 1.1, 1.2, inner_radius);

      CHECK(shape_and_size.at(shape_name)->time_bounds() ==
            std::array{0.2, 1.1});
      CHECK(shape_and_size.at(size_name)->time_bounds() ==
            std::array{0.2, 1.2});

      const auto shape = shape_and_size.at(shape_name)->func_and_2_derivs(0.2);
      const auto expected_shape = functions_of_time.at(shape_name)
                                      ->create_at_time(0.2, 1.1)
                                      ->func_and_2_derivs(0.2);

      // We restricted from LMax = 8 to LMax = 6
      const size_t expected_l_max = l_max - 2;
      const size_t expected_size =
          ylm::Spherepack::spectral_size(expected_l_max, expected_l_max);
      ylm::SpherepackIterator file_iter{l_max, l_max};
      ylm::SpherepackIterator iter{expected_l_max, expected_l_max};

      for (size_t i = 0; i < 3; i++) {
        CAPTURE(i);
        CHECK(gsl::at(shape, i).size() == expected_size);

        // Loop pointwise so we only check the coefficients that matter
        for (size_t l = 0; l <= expected_l_max; l++) {
          CAPTURE(l);
          for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
            CAPTURE(m);
            CHECK(gsl::at(shape, i)[iter.set(l, m)()] ==
                  gsl::at(expected_shape, i)[file_iter.set(l, m)()]);
          }
        }
      }
      CHECK(shape_and_size.at(size_name)->func_and_2_derivs(0.2) ==
            functions_of_time.at(size_name)->func_and_2_derivs(0.2));
    }

    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }
  }
  {
    INFO("YlmsFromSpec");
    const std::string test_filename{"PartialEclipseOfTheHeart.dat"};
    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }
    const double time = 1.7;
    // Purposefully larger than the LMax in the options so that the
    // Strahlkorpers will be restricted
    const size_t file_l_max = 10;
    const std::uniform_real_distribution<double> distribution(0.1, 2.0);
    using Frame = ::Frame::Inertial;
    ylm::Strahlkorper<Frame> strahlkorper{};
    std::vector<std::string> legend{};
    std::vector<double> data{};

    // Scoped to close dat file
    {
      std::fstream test_file{};
      test_file.open(test_filename, std::ios_base::out);
      test_file << std::scientific << std::setprecision(16);
      legend.clear();
      data.clear();
      const auto radius = make_with_random_values<DataVector>(
          generator, distribution,
          ylm::Spherepack::physical_size(file_l_max, file_l_max));
      strahlkorper = ylm::Strahlkorper<Frame>(file_l_max, file_l_max, radius,
                                              std::array{0.0, 0.0, 0.0});

      ylm::fill_ylm_legend_and_data(make_not_null(&legend),
                                    make_not_null(&(data)), strahlkorper, time,
                                    file_l_max);

      test_file << "# This is a random comment\n";
      // These columns won't be exactly what is in the SpEC dat file, but they
      // will be close enough. Also SpEC starts columns numbers from 1
      size_t col_num = 1;
      for (const std::string& col : legend) {
        // SpEC doesn't write Lmax
        if (col == "Lmax") {
          continue;
        }

        test_file << "# [" << col_num << "] = " << col << "\n";
        col_num++;
      }
      test_file << "# Another random comment to show\n";

      // Write three rows of same data
      for (size_t i = 0; i < 3; i++) {
        // Time and center
        test_file << time + static_cast<double>(i) << " " << 0.0 << " " << 0.0
                  << " " << 0.0;
        // Skip writing time, center, and Lmax
        for (size_t j = 5; j < data.size(); j++) {
          test_file << " " << data[j];
        }
        test_file << "\n";
      }
    }

    {
      const auto shape_map_options = TestHelpers::test_option_tag<
          domain::creators::time_dependent_options::ShapeMapOptions<
              false, domain::ObjectLabel::None>>(
          "LMax: 8\n"
          "CoefficientTruncationLimit: 0.0\n"
          "InitialValues:\n"
          "  DatFilename: PartialEclipseOfTheHeart.dat\n"
          "  MatchTime: 2.7\n"
          "  MatchTimeEpsilon: Auto\n"
          "  SetL1CoefsToZero: True\n"
          "SizeInitialValues: Auto");

      REQUIRE(shape_map_options.has_value());
      REQUIRE(std::holds_alternative<
              ShapeMapOptions<false, domain::ObjectLabel::None>>(
          shape_map_options.value()));

      const FunctionsOfTimeMap shape_and_size = get_shape_and_size(
          shape_map_options.value(), 0.2, 1.1, 1.2, inner_radius);

      const auto shape_funcs =
          shape_and_size.at("Shape")->func_and_2_derivs(0.2);
      const auto size_funcs = shape_and_size.at("Size")->func_and_2_derivs(0.2);

      ylm::SpherepackIterator iter{l_max, l_max};
      ylm::SpherepackIterator file_iter{file_l_max, file_l_max};
      for (size_t i = 0; i < shape_funcs.size(); i++) {
        const size_t expected_size =
            ylm::Spherepack::spectral_size(l_max, l_max);
        // Make sure they were restricted properly
        CHECK(gsl::at(shape_funcs, i).size() == expected_size);

        if (i > 0) {
          CHECK(gsl::at(shape_funcs, i) == DataVector{expected_size, 0.0});
          continue;
        }

        // Loop pointwise so we only check the coefficients that matter
        for (size_t l = 0; l <= l_max; l++) {
          for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
            CAPTURE(l);
            CAPTURE(m);
            const double expected_value =
                l < 2
                    ? 0.0
                    : -1.0 * strahlkorper.coefficients()[file_iter.set(l, m)()];
            CHECK(gsl::at(shape_funcs, i)[iter.set(l, m)()] == expected_value);
          }
        }
      }
      CHECK(size_funcs ==
            std::array{DataVector{-1.0 * strahlkorper.coefficients()[0] *
                                  sqrt(0.5 * M_PI)},
                       DataVector{0.0}, DataVector{0.0}});
    }

    {
      const auto shape_map_options = TestHelpers::test_option_tag<
          domain::creators::time_dependent_options::ShapeMapOptions<
              false, domain::ObjectLabel::None>>(
          "LMax: 8\n"
          "CoefficientTruncationLimit: 0.0\n"
          "InitialValues:\n"
          "  DatFilename: PartialEclipseOfTheHeart.dat\n"
          "  MatchTime: 10000.0\n"
          "  MatchTimeEpsilon: Auto\n"
          "  SetL1CoefsToZero: True\n"
          "SizeInitialValues: Auto");
      CHECK_THROWS_WITH(
          (get_shape_and_size(shape_map_options.value(), 0.2, 1.1, 1.2,
                              inner_radius)),
          Catch::Matchers::ContainsSubstring(
              "Unable to find requested time 1") and
              Catch::Matchers::ContainsSubstring(
                  "in SpEC dat file PartialEclipseOfTheHeart.dat"));
    }

    if (file_system::check_if_file_exists(test_filename)) {
      file_system::rm(test_filename, true);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependentOptions.ShapeMap",
                  "[Domain][Unit]") {
  domain::FunctionsOfTime::register_derived_with_charm();
  MAKE_GENERATOR(generator);
  test_kerr_schild_boyer_lindquist();
  test_ylms_from_file();
  test_funcs(make_not_null(&generator));
}
}  // namespace domain::creators::time_dependent_options
