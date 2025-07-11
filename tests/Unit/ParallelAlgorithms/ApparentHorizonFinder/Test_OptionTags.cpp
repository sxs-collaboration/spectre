// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/OptionTags.hpp"

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.OptionTags",
                  "[ApparentHorizonFinder][Unit]") {
  // Constants used in this test.
  const size_t l_max = 12;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.05, 0.06, 0.07}};

  // Options for ApparentHorizon
  ah::HorizonOptions<::Frame::Grid> apparent_horizon_opts(
      ylm::Strahlkorper<Frame::Grid>{l_max, radius, center}, FastFlow{},
      Verbosity::Verbose, 3_st, std::nullopt);

  // Test creation of options
  const auto created_opts =
      TestHelpers::test_creation<ah::HorizonOptions<Frame::Grid>>(
          "FastFlow:\n"
          "  Flow: Fast\n"
          "  Alpha: 1.0\n"
          "  Beta: 0.5\n"
          "  AbsTol: 1e-12\n"
          "  TruncationTol: 1e-2\n"
          "  DivergenceTol: 1.2\n"
          "  DivergenceIter: 5\n"
          "  MaxIts: 100\n"
          "Verbosity: Verbose\n"
          "InitialGuess:\n"
          "  Center: [0.05, 0.06, 0.07]\n"
          "  Radius: 2.0\n"
          "  LMax: 12\n"
          "MaxInterpolationRetries: 3\n"
          "BlocksForHorizonFind: All");
  CHECK(created_opts == apparent_horizon_opts);
}
