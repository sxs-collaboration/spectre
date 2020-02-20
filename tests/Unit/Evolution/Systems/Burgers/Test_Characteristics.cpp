// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Characteristics.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Burgers.Characteristics", "[Unit][Burgers]") {
  TestHelpers::db::test_compute_tag<Burgers::Tags::CharacteristicSpeedsCompute>(
      "CharacteristicSpeeds");
  {
    const auto box = db::create<
        db::AddSimpleTags<Burgers::Tags::U, Tags::UnnormalizedFaceNormal<1>>,
        db::AddComputeTags<Burgers::Tags::CharacteristicSpeedsCompute>>(
        Scalar<DataVector>{{{{4.0}}}}, tnsr::i<DataVector, 1>{{{{1.0}}}});
    CHECK(db::get<Burgers::Tags::CharacteristicSpeedsCompute>(box)[0] == 4.0);
  }
  {
    const auto box = db::create<
        db::AddSimpleTags<Burgers::Tags::U, Tags::UnnormalizedFaceNormal<1>>,
        db::AddComputeTags<Burgers::Tags::CharacteristicSpeedsCompute>>(
        Scalar<DataVector>{{{{4.0}}}}, tnsr::i<DataVector, 1>{{{{-1.0}}}});
    CHECK(db::get<Burgers::Tags::CharacteristicSpeedsCompute>(box)[0] == -4.0);
  }
}

SPECTRE_TEST_CASE("Unit.Burgers.ComputeLargestCharacteristicSpeed",
                  "[Unit][Burgers]") {
  CHECK(Burgers::ComputeLargestCharacteristicSpeed::apply(
            Scalar<DataVector>{{{{1., 2., 4., 3.}}}}) == 4.);
  CHECK(Burgers::ComputeLargestCharacteristicSpeed::apply(
            Scalar<DataVector>{{{{1., 2., 4., -5.}}}}) == 5.);
}
