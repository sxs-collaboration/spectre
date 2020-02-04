// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>

#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Literals.hpp"
#include "tests/Unit/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "tests/Unit/TestCreation.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.OptionTags", "[Unit][Cce]") {
  CHECK(TestHelpers::test_creation<size_t, Cce::OptionTags::LMax>("8") == 8_st);
  CHECK(TestHelpers::test_creation<size_t, Cce::OptionTags::FilterLMax>("7") ==
        7_st);
  CHECK(TestHelpers::test_creation<double, Cce::OptionTags::RadialFilterAlpha>(
            "32.5") == 32.5);
  CHECK(TestHelpers::test_creation<size_t,
                                   Cce::OptionTags::RadialFilterHalfPower>(
            "20") == 20_st);
  CHECK(TestHelpers::test_creation<size_t, Cce::OptionTags::ObservationLMax>(
            "6") == 6_st);
  CHECK(
      TestHelpers::test_creation<size_t, Cce::OptionTags::NumberOfRadialPoints>(
          "3") == 3_st);
  CHECK(TestHelpers::test_creation<double, Cce::OptionTags::EndTime>("4.0") ==
        4.0);
  CHECK(TestHelpers::test_creation<double, Cce::OptionTags::StartTime>("2.0") ==
        2.0);
  CHECK(TestHelpers::test_creation<double, Cce::OptionTags::TargetStepSize>(
            "0.5") == 0.5);
  CHECK(TestHelpers::test_creation<std::string,
                                   Cce::OptionTags::BoundaryDataFilename>(
            "OptionTagsCceR0100.h5") == "OptionTagsCceR0100.h5");
  CHECK(TestHelpers::test_creation<size_t, Cce::OptionTags::H5LookaheadTimes>(
            "5") == 5_st);
  CHECK(TestHelpers::test_creation<size_t,
                                   Cce::OptionTags::ScriInterpolationOrder>(
            "4") == 4_st);
  const std::string filename = "OptionTagsTestCceR0100.h5";
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  Cce::TestHelpers::write_test_file(
      gr::Solutions::KerrSchild{1.0, {{0.2, 0.2, 0.2}}, {{0.0, 0.0, 0.0}}},
      filename, 4.0, 100.0, 0.0, 0.1, 8);

  CHECK(
      Cce::InitializationTags::H5WorldtubeBoundaryDataManager::
          create_from_options(8, filename, 3,
                              std::make_unique<intrp::CubicSpanInterpolator>())
              .get_l_max() == 8);

  CHECK(Cce::InitializationTags::LMax::create_from_options(8u) == 8u);
  CHECK(Cce::InitializationTags::NumberOfRadialPoints::create_from_options(
            6u) == 6u);

  CHECK(Cce::InitializationTags::StartTime::create_from_options(
            -std::numeric_limits<double>::infinity(),
            "OptionTagsTestCceR0100.h5") == 2.5);
  CHECK(Cce::InitializationTags::StartTime::create_from_options(
            3.3, "OptionTagsTestCceR0100.h5") == 3.3);

  CHECK(Cce::InitializationTags::TargetStepSize::create_from_options(0.2) ==
        0.2);

  CHECK(Cce::InitializationTags::EndTime::create_from_options(
            std::numeric_limits<double>::infinity(),
            "OptionTagsTestCceR0100.h5") == 5.4);
  CHECK(Cce::InitializationTags::EndTime::create_from_options(
            2.2, "OptionTagsTestCceR0100.h5") == 2.2);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
