// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Info.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Amr.Info", "[Domain][Unit]") {
  amr::Info<1> info_0{std::array{amr::Flag::Join},
                      Mesh<1>{3_st, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto}};
  amr::Info<1> info_1{std::array{amr::Flag::Split},
                      Mesh<1>{3_st, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto}};
  amr::Info<1> info_2{std::array{amr::Flag::Split},
                      Mesh<1>{4_st, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto}};
  test_serialization(info_0);
  CHECK(info_0 != info_1);
  CHECK(info_1 != info_2);
  CHECK(info_0 != info_2);
  std::string expected_output = MakeString{}
                                << "Flags: " << info_0.flags
                                << " New mesh: " << info_0.new_mesh;
  CHECK(get_output(info_0) == expected_output);
}
