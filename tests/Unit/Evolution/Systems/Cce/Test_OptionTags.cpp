// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/BouncingBlackHole.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/GaugeWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/LinearizedBondiSachs.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/RotatingSchwarzschild.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/TeukolskyWave.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Initialize/ConformalFactor.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"
#include "Evolution/Systems/Cce/Initialize/NoIncomingRadiation.hpp"
#include "Evolution/Systems/Cce/Initialize/ZeroNonSmooth.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/RungeKutta3.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TypeTraits.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.OptionTags", "[Unit][Cce]") {
  TestHelpers::db::test_simple_tag<
      Cce::InitializationTags::ScriInterpolationOrder>(
      "ScriInterpolationOrder");
  TestHelpers::db::test_simple_tag<Cce::InitializationTags::ExtractionRadius>(
      "ExtractionRadius");
  TestHelpers::db::test_simple_tag<Cce::InitializationTags::ScriOutputDensity>(
      "ScriOutputDensity");
  TestHelpers::db::test_simple_tag<Cce::Tags::H5WorldtubeBoundaryDataManager>(
      "H5WorldtubeBoundaryDataManager");
  TestHelpers::db::test_simple_tag<Cce::Tags::FilePrefix>("FilePrefix");
  TestHelpers::db::test_simple_tag<Cce::Tags::LMax>("LMax");
  TestHelpers::db::test_simple_tag<Cce::Tags::NumberOfRadialPoints>(
      "NumberOfRadialPoints");
  TestHelpers::db::test_simple_tag<Cce::Tags::ObservationLMax>(
      "ObservationLMax");
  TestHelpers::db::test_simple_tag<Cce::Tags::FilterLMax>("FilterLMax");
  TestHelpers::db::test_simple_tag<Cce::Tags::RadialFilterAlpha>(
      "RadialFilterAlpha");
  TestHelpers::db::test_simple_tag<Cce::Tags::RadialFilterHalfPower>(
      "RadialFilterHalfPower");
  TestHelpers::db::test_simple_tag<Cce::Tags::StartTimeFromFile>(
      "StartTimeFromFile");
  TestHelpers::db::test_simple_tag<Cce::Tags::EndTimeFromFile>(
      "EndTimeFromFile");
  TestHelpers::db::test_simple_tag<Cce::Tags::NoEndTime>("NoEndTime");
  TestHelpers::db::test_simple_tag<Cce::Tags::SpecifiedStartTime>(
      "SpecifiedStartTime");
  TestHelpers::db::test_simple_tag<Cce::Tags::SpecifiedEndTime>(
      "SpecifiedEndTime");
  TestHelpers::db::test_simple_tag<Cce::Tags::GhInterfaceManager>(
      "GhInterfaceManager");
  TestHelpers::db::test_simple_tag<Cce::Tags::AnalyticBoundaryDataManager>(
      "AnalyticBoundaryDataManager");
  TestHelpers::db::test_simple_tag<Cce::Tags::InitializeJ<true>>("InitializeJ");
  TestHelpers::db::test_simple_tag<Cce::Tags::InitializeJ<false>>(
      "InitializeJ");
  TestHelpers::db::test_simple_tag<Cce::Tags::AnalyticInitializeJ>(
      "AnalyticInitializeJ");
  TestHelpers::db::test_simple_tag<Cce::Tags::OutputNoninertialNews>(
      "OutputNoninertialNews");
  TestHelpers::db::test_simple_tag<
      Cce::Tags::CceEvolutionPrefix<::Tags::TimeStepper<TimeStepper>>>(
      "TimeStepper");

  CHECK(
      TestHelpers::test_option_tag<Cce::OptionTags::BondiSachsOutputFilePrefix>(
          "Shrek") == "Shrek");
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::LMax>("8") == 8_st);
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::FilterLMax>("7") == 7_st);
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::RadialFilterAlpha>(
            "32.5") == 32.5);
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::RadialFilterHalfPower>(
            "20") == 20_st);
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::ObservationLMax>("6") ==
        6_st);
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::NumberOfRadialPoints>(
            "3") == 3_st);
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::ExtractionRadius>(
            "100.0") == 100.0);

  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::EndTime>("4.0") ==
        Options::Auto<double>{4.0});
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::EndTime>("Auto") ==
        Options::Auto<double>{});
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::StartTime>("2.0") ==
        Options::Auto<double>{2.0});
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::StartTime>("Auto") ==
        Options::Auto<double>{});
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::BoundaryDataFilename>(
            "OptionTagsCceR0100.h5") == "OptionTagsCceR0100.h5");
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::H5LookaheadTimes>("5") ==
        5_st);
  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::ScriInterpolationOrder>(
            "4") == 4_st);

  CHECK(TestHelpers::test_option_tag<Cce::OptionTags::ScriOutputDensity>("6") ==
        6_st);

  TestHelpers::test_option_tag<Cce::OptionTags::InitializeJ<true>>(
      "InverseCubic");
  TestHelpers::test_option_tag<Cce::OptionTags::InitializeJ<false>>(
      "InverseCubic");
  TestHelpers::test_option_tag<Cce::OptionTags::AnalyticSolution>(
      "BouncingBlackHole:\n"
      "  Period: 40.0\n"
      "  ExtractionRadius: 30.0\n"
      "  Mass: 1.0\n"
      "  Amplitude: 2.0");
  TestHelpers::test_option_tag<Cce::OptionTags::AnalyticSolution>(
      "GaugeWave:\n"
      "  ExtractionRadius: 40.0\n"
      "  Mass: 1.0\n"
      "  Frequency: 0.5\n"
      "  Amplitude: 0.1\n"
      "  PeakTime: 50.0\n"
      "  Duration: 10.0");
  TestHelpers::test_option_tag<Cce::OptionTags::AnalyticSolution>(
      "LinearizedBondiSachs:\n"
      "  ExtractionRadius: 40.0\n"
      "  InitialModes: [[0.20, 0.10], [0.08, 0.04]]\n"
      "  Frequency: 0.2");
  TestHelpers::test_option_tag<Cce::OptionTags::AnalyticSolution>(
      "RobinsonTrautman:\n"
      "  InitialModes:\n"
      "    - [0.0, 0.0]\n"
      "    - [0.0, 0.0]\n"
      "    - [0.0, 0.0]\n"
      "    - [0.0, 0.0]\n"
      "    - [1.0, 0.5]\n"
      "  ExtractionRadius: 20.0\n"
      "  LMax: 16\n"
      "  Tolerance: 1e-10\n"
      "  StartTime: 0.0");
  TestHelpers::test_option_tag<Cce::OptionTags::AnalyticSolution>(
      "RotatingSchwarzschild:\n"
      "  ExtractionRadius: 20.0\n"
      "  Mass: 1.0\n"
      "  Frequency: 0.0");
  TestHelpers::test_option_tag<Cce::OptionTags::AnalyticSolution>(
      "TeukolskyWave:\n"
      "  ExtractionRadius: 40.0\n"
      "  Amplitude: 0.1\n"
      "  Duration: 3.0");
  TestHelpers::test_option_tag_factory_creation<
      Cce::OptionTags::CceEvolutionPrefix<
          ::OptionTags::TimeStepper<TimeStepper>>,
      TimeSteppers::RungeKutta3>("RungeKutta3:");

  const std::string filename = "OptionTagsTestCceR0100.h5";
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  Cce::TestHelpers::write_test_file(
      gr::Solutions::KerrSchild{1.0, {{0.2, 0.2, 0.2}}, {{0.0, 0.0, 0.0}}},
      filename, 4.0, 100.0, 0.0, 0.1, 8);

  CHECK(Cce::Tags::H5WorldtubeBoundaryDataManager::create_from_options(
            8, filename, 3, std::make_unique<intrp::CubicSpanInterpolator>(),
            false, true, std::nullopt)
            ->get_l_max() == 8);

  CHECK(Cce::Tags::FilePrefix::create_from_options("Shrek 2") == "Shrek 2");
  CHECK(Cce::Tags::LMax::create_from_options(8u) == 8u);
  CHECK(Cce::Tags::NumberOfRadialPoints::create_from_options(6u) == 6u);

  CHECK(Cce::Tags::StartTimeFromFile::create_from_options(
            std::optional<double>{}, "OptionTagsTestCceR0100.h5", false) ==
        2.5);
  CHECK(Cce::Tags::StartTimeFromFile::create_from_options(
            std::optional<double>{3.3}, "OptionTagsTestCceR0100.h5", false) ==
        3.3);
  CHECK(Cce::Tags::SpecifiedStartTime::create_from_options(
            std::optional<double>(2.0)) == 2.0);

  CHECK(Cce::Tags::EndTimeFromFile::create_from_options(
            std::optional<double>{}, "OptionTagsTestCceR0100.h5", false) ==
        5.4);
  CHECK(Cce::Tags::EndTimeFromFile::create_from_options(
            std::optional<double>{2.2}, "OptionTagsTestCceR0100.h5", false) ==
        2.2);
  CHECK(Cce::Tags::SpecifiedEndTime::create_from_options(
            std::optional<double>(40.0)) == 40.0);

  CHECK(Cce::Tags::ObservationLMax::create_from_options(5_st) == 5_st);

  CHECK(Cce::InitializationTags::ScriInterpolationOrder::create_from_options(
            6_st) == 6_st);

  CHECK(Cce::InitializationTags::ScriOutputDensity::create_from_options(4_st) ==
        4_st);

  CHECK(Cce::Tags::AnalyticBoundaryDataManager::create_from_options(
            10.0, 8,
            std::make_unique<Cce::Solutions::RotatingSchwarzschild>(10.0, 1.0,
                                                                    0.5))
            .get_l_max() == 8);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
