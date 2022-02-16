// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Framework/TestCreation.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <evolution::dg::subcell::fd::ReconstructionMethod ReconsMethod>
void test_construct_from_options() {
  const auto created = TestHelpers::test_creation<
      evolution::dg::subcell::fd::ReconstructionMethod>(
      get_output(ReconsMethod));
  CHECK(created == ReconsMethod);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Fd.ReconstructionMethod",
                  "[Evolution][Unit]") {
  using evolution::dg::subcell::fd::ReconstructionMethod;
  CHECK(get_output(ReconstructionMethod::DimByDim) == "DimByDim");
  CHECK(get_output(ReconstructionMethod::AllDimsAtOnce) == "AllDimsAtOnce");

  test_construct_from_options<ReconstructionMethod::DimByDim>();
  test_construct_from_options<ReconstructionMethod::AllDimsAtOnce>();
}
