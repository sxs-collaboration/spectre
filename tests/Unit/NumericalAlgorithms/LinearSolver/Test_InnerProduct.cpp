// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <blaze/math/DynamicVector.h>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename DataType>
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct AnotherScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataType>;
};

}  // namespace

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.LinearSolver.InnerProduct",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  {
    INFO("Blaze vector");
    const blaze::DynamicVector<double> lhs{1., 0., 2.};
    const blaze::DynamicVector<double> rhs{1.5, 1., 3.};
    CHECK(LinearSolver::inner_product(lhs, rhs) == dot(lhs, rhs));
    CHECK(LinearSolver::magnitude_square(lhs) == 5.);
  }
  {
    INFO("Variables");
    const Variables<tmpl::list<ScalarFieldTag<DataVector>>> vars{3, 1.};
    const Variables<tmpl::list<AnotherScalarFieldTag<DataVector>>> other_vars{
        3, 2.};
    CHECK(LinearSolver::inner_product(vars, other_vars) == 6.);
    CHECK(LinearSolver::magnitude_square(vars) == 3.);
  }
  {
    INFO("Complex Variables");
    const Variables<tmpl::list<ScalarFieldTag<ComplexDataVector>>> vars{
        3, std::complex<double>(1., 2.)};
    const Variables<tmpl::list<AnotherScalarFieldTag<ComplexDataVector>>>
        other_vars{3, std::complex<double>(2., 3.)};
    CHECK(LinearSolver::inner_product(vars, other_vars) ==
          std::complex<double>(24., -3.));
    CHECK(LinearSolver::magnitude_square(vars) == 15.);
  }
}
