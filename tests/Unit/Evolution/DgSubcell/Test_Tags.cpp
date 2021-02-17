// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/TciGridHistory.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

/// \cond
class DataVector;
/// \endcond

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, 3, Frame::Inertial>;
};

template <size_t Dim>
void test() {
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::Mesh<Dim>>(
      "Subcell(Mesh)");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Tags",
                  "[Evolution][Unit]") {
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::ActiveGrid>(
      "ActiveGrid");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::Inactive<Var1>>("Inactive(Var1)");
  TestHelpers::db::test_simple_tag<evolution::dg::subcell::Tags::Inactive<
      ::Tags::Variables<tmpl::list<Var1, Var2>>>>(
      "Inactive(Variables(Var1,Var2))");
  TestHelpers::db::test_simple_tag<
      evolution::dg::subcell::Tags::TciGridHistory>("TciGridHistory");

  test<1>();
  test<2>();
  test<3>();
}
