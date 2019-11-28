// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace hydro {
namespace {
template <size_t Dim, typename Frame, typename DataType>
void test_lorentz_factor(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(&lorentz_factor<DataType, Dim, Frame>,
                                    "TestFunctions", "lorentz_factor",
                                    {{{0.0, 1.0 / sqrt(Dim)}}}, used_for_size);
  pypp::check_with_random_values<1>(&lorentz_factor<DataType, Dim, Frame>,
                                    "TestFunctions", "lorentz_factor",
                                    {{{-1.0 / sqrt(Dim), 0.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.LorentzFactor",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  const DataVector dv(5);
  test_lorentz_factor<1, Frame::System>(dv);
  test_lorentz_factor<1, Frame::GlobalTimeIndependent>(dv);
  test_lorentz_factor<2, Frame::System>(dv);
  test_lorentz_factor<2, Frame::GlobalTimeIndependent>(dv);
  test_lorentz_factor<3, Frame::System>(dv);
  test_lorentz_factor<3, Frame::GlobalTimeIndependent>(dv);

  test_lorentz_factor<1, Frame::System>(0.0);
  test_lorentz_factor<1, Frame::GlobalTimeIndependent>(0.0);
  test_lorentz_factor<2, Frame::System>(0.0);
  test_lorentz_factor<2, Frame::GlobalTimeIndependent>(0.0);
  test_lorentz_factor<3, Frame::System>(0.0);
  test_lorentz_factor<3, Frame::GlobalTimeIndependent>(0.0);

  // Check compute item works correctly in DataBox
  CHECK(Tags::LorentzFactorCompute<DataVector, 2, Frame::System>::name() ==
        "LorentzFactor");
  tnsr::i<DataVector, 2> velocity_one_form{
      {{DataVector{5, 0.2}, DataVector{5, 0.3}}}};
  tnsr::I<DataVector, 2> velocity{{{DataVector{5, 0.25}, DataVector{5, 0.35}}}};
  const auto box = db::create<
      db::AddSimpleTags<
          Tags::SpatialVelocity<DataVector, 2, Frame::System>,
          Tags::SpatialVelocityOneForm<DataVector, 2, Frame::System>>,
      db::AddComputeTags<
          Tags::LorentzFactorCompute<DataVector, 2, Frame::System>>>(
      velocity, velocity_one_form);
  CHECK(db::get<Tags::LorentzFactor<DataVector>>(box) ==
        lorentz_factor(velocity, velocity_one_form));
}
}  // namespace hydro
