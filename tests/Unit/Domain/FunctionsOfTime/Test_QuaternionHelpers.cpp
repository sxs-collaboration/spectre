// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/math/quaternion.hpp>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/QuaternionHelpers.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.QuaternionHelpers",
                  "[Unit][Domain]") {
  using quaternion = boost::math::quaternion<double>;

  quaternion quat1(1.0, 2.0, 3.0, -4.0);
  DataVector dv{1.0, 2.0, 3.0, -4.0};
  DataVector dv_short{9.9, 9.8, 9.7};

  DataVector dv2 = quaternion_to_datavector(quat1);
  CHECK(dv2 == dv);

  quaternion quat2 = datavector_to_quaternion(dv2);
  CHECK(quat2 == quat1);

  quaternion quat3 = datavector_to_quaternion(dv_short);
  CHECK(quat3 == quaternion{0.0, 9.9, 9.8, 9.7});

  normalize_quaternion(make_not_null(&quat2));
  CHECK(norm(quat2) == approx(1.0));
  CHECK(abs(quat2) == approx(1.0));
}
