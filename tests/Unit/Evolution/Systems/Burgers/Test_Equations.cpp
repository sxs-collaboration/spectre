// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Equations.hpp"

SPECTRE_TEST_CASE("Unit.Burgers.ComputeLargestCharacteristicSpeed",
                  "[Unit][Burgers]") {
  CHECK(Burgers::ComputeLargestCharacteristicSpeed::apply(
            Scalar<DataVector>{{{{1., 2., 4., 3.}}}}) == 4.);
  CHECK(Burgers::ComputeLargestCharacteristicSpeed::apply(
            Scalar<DataVector>{{{{1., 2., 4., -5.}}}}) == 5.);
}
