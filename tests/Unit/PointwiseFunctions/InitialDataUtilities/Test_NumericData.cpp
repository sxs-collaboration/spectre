// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"

template <typename Subclass>
void test_numeric_data() {
  const std::string file_name =
      unit_test_src_path() + "/Visualization/Python/VolTestData*.h5";
  const std::string option_string = "FileGlob: " + file_name +
                                    "\nSubgroup: element_data\n"
                                    "ObservationStep: -1\n"
                                    "ExtrapolateIntoExcisions: False\n";
  const auto created = TestHelpers::test_creation<Subclass>(option_string);
  const elliptic::analytic_data::NumericData numeric_data{
      file_name, "element_data", -1, false};
  CHECK(created == numeric_data);
  test_serialization(numeric_data);
  test_copy_semantics(numeric_data);

  const auto interpolated_data = numeric_data.variables(
      tnsr::I<DataVector, 3>{
          {{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}}}},
      tmpl::list<ScalarWave::Tags::Psi, ScalarWave::Tags::Phi<3>>{});
  const auto& psi = get(get<ScalarWave::Tags::Psi>(interpolated_data));
  CHECK(psi[0] == approx(-0.07059806932542323));
  CHECK(psi[1] == approx(0.7869554122196492));
  CHECK(psi[2] == approx(0.9876185584100299));
  const auto& phi_y = get<1>(get<ScalarWave::Tags::Phi<3>>(interpolated_data));
  CHECK(phi_y[0] == approx(1.0569673471948728));
  CHECK(phi_y[1] == approx(0.6741524090220188));
  CHECK(phi_y[2] == approx(0.2629752479142838));
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.InitialDataUtilities.NumericData",
                  "[Unit][PointwiseFunctions]") {
  test_numeric_data<elliptic::analytic_data::NumericData>();
  test_numeric_data<evolution::initial_data::NumericData>();
}
