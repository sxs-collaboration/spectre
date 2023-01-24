// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/Punctures/AdmIntegrals.hpp"

namespace Punctures {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Punctures.AdmIntegrals",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Punctures"};
  const DataVector used_for_size{5};
  pypp::check_with_random_values<1>(
      static_cast<void (*)(gsl::not_null<Scalar<DataVector>*>,
                           const Scalar<DataVector>&, const Scalar<DataVector>&,
                           const Scalar<DataVector>&)>(&adm_mass_integrand),
      "AdmIntegrals", {"adm_mass_integrand"}, {{{-1., 1.}}}, used_for_size);

  TestHelpers::db::test_simple_tag<Tags::AdmMassIntegrand>("AdmMassIntegrand");
  TestHelpers::db::test_compute_tag<Tags::AdmMassIntegrandCompute>(
      "AdmMassIntegrand");
}

}  // namespace Punctures
