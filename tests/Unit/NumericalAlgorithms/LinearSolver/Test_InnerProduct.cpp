// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <blaze/math/DynamicVector.h>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct AnotherScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.LinearSolver.InnerProduct",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  const blaze::DynamicVector<double> lhs{1., 0., 2.};
  const blaze::DynamicVector<double> rhs{1.5, 1., 3.};
  CHECK(LinearSolver::inner_product(lhs, rhs) == dot(lhs, rhs));

  const Variables<tmpl::list<ScalarFieldTag>> vars{3, 1.};
  const Variables<tmpl::list<AnotherScalarFieldTag>> other_vars{3, 2.};
  CHECK(LinearSolver::inner_product(vars, other_vars) == 6.);
}
