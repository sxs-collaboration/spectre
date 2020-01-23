// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
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
            "CceR0100.h5") == "CceR0100.h5");
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

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
