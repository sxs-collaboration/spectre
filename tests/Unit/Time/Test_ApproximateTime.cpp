// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Time/ApproximateTime.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Time.ApproximateTime", "[Unit][Time]") {
  const double val1 = 1.25;
  const double val2 = 3.5;
  const Slab slab(val1, val2);

  const Time time1 = slab.start();
  const Time time2 = slab.end();
  const ApproximateTime approx1{val1};
  const ApproximateTime approx2{val2};
  const TimeDelta diff = time2 - time1;
  const ApproximateTimeDelta approx_diff{val2 - val1};
  const TimeDelta neg_diff = time1 - time2;
  const ApproximateTimeDelta approx_neg_diff{val1 - val2};

  CHECK(approx1.value() == time1.value());
  CHECK(approx2.value() == time2.value());
  CHECK(approx_diff.value() == diff.value());
  CHECK(approx_neg_diff.value() == neg_diff.value());

  CHECK(approx1 == approx1);
  CHECK(approx1 == time1);
  CHECK(time1 == approx1);
  CHECK_FALSE(approx1 == approx2);
  CHECK_FALSE(approx1 == time2);
  CHECK_FALSE(time2 == approx1);
  CHECK_FALSE(approx1 != approx1);
  CHECK_FALSE(approx1 != time1);
  CHECK_FALSE(time1 != approx1);
  CHECK(approx1 != approx2);
  CHECK(approx1 != time2);
  CHECK(time2 != approx1);

  CHECK(approx_diff == approx_diff);
  CHECK(approx_diff == diff);
  CHECK(diff == approx_diff);
  CHECK_FALSE(approx_diff == approx_neg_diff);
  CHECK_FALSE(approx_diff == neg_diff);
  CHECK_FALSE(neg_diff == approx_diff);
  CHECK_FALSE(approx_diff != approx_diff);
  CHECK_FALSE(approx_diff != diff);
  CHECK_FALSE(diff != approx_diff);
  CHECK(approx_diff != approx_neg_diff);
  CHECK(approx_diff != neg_diff);
  CHECK(neg_diff != approx_diff);

  CHECK(approx1 < approx2);
  CHECK(approx1 < time2);
  CHECK(time1 < approx2);
  CHECK_FALSE(approx2 < approx1);
  CHECK_FALSE(approx2 < time1);
  CHECK_FALSE(time2 < approx1);
  CHECK_FALSE(approx1 < approx1);
  CHECK_FALSE(approx1 < time1);
  CHECK_FALSE(time1 < approx1);

  CHECK_FALSE(approx1 > approx2);
  CHECK_FALSE(approx1 > time2);
  CHECK_FALSE(time1 > approx2);
  CHECK(approx2 > approx1);
  CHECK(approx2 > time1);
  CHECK(time2 > approx1);
  CHECK_FALSE(approx1 > approx1);
  CHECK_FALSE(approx1 > time1);
  CHECK_FALSE(time1 > approx1);

  CHECK(approx1 <= approx2);
  CHECK(approx1 <= time2);
  CHECK(time1 <= approx2);
  CHECK_FALSE(approx2 <= approx1);
  CHECK_FALSE(approx2 <= time1);
  CHECK_FALSE(time2 <= approx1);
  CHECK(approx1 <= approx1);
  CHECK(approx1 <= time1);
  CHECK(time1 <= approx1);

  CHECK_FALSE(approx1 >= approx2);
  CHECK_FALSE(approx1 >= time2);
  CHECK_FALSE(time1 >= approx2);
  CHECK(approx2 >= approx1);
  CHECK(approx2 >= time1);
  CHECK(time2 >= approx1);
  CHECK(approx1 >= approx1);
  CHECK(approx1 >= time1);
  CHECK(time1 >= approx1);

  CHECK(+approx_diff == approx_diff);
  CHECK(-approx_diff == approx_neg_diff);
  CHECK(approx_diff.is_positive());
  CHECK(not approx_neg_diff.is_positive());
  CHECK(abs(approx_diff) == approx_diff);
  CHECK(abs(approx_neg_diff) == approx_diff);

  CHECK(approx1 - approx2 == time1 - time2);
  CHECK(approx1 - time2 == time1 - time2);
  CHECK(time1 - approx2 == time1 - time2);

  CHECK(approx1 + approx_diff == time1 + diff);
  CHECK(approx1 + diff == time1 + diff);
  CHECK(time1 + approx_diff == time1 + diff);

  CHECK(approx_diff + approx1 == diff + time1);
  CHECK(approx_diff + approx1 == diff + time1);
  CHECK(approx_diff + time1 == diff + time1);

  CHECK(approx2 - approx_diff == time2 - diff);
  CHECK(approx2 - diff == time2 - diff);
  CHECK(time2 - approx_diff == time2 - diff);

  CHECK(approx_diff + approx_diff == diff + diff);
  CHECK(approx_diff + diff == diff + diff);
  CHECK(diff + approx_diff == diff + diff);
  CHECK(approx_diff + approx_neg_diff == diff + neg_diff);
  CHECK(approx_diff + neg_diff == diff + neg_diff);
  CHECK(diff + approx_neg_diff == diff + neg_diff);

  CHECK(approx_diff - approx_diff == diff - diff);
  CHECK(approx_diff - diff == diff - diff);
  CHECK(diff - approx_diff == diff - diff);
  CHECK(approx_diff - approx_neg_diff == diff - neg_diff);
  CHECK(approx_diff - neg_diff == diff - neg_diff);
  CHECK(diff - approx_neg_diff == diff - neg_diff);

  CHECK(approx_diff / approx_diff == diff / diff);
  CHECK(approx_diff / diff == diff / diff);
  CHECK(diff / approx_diff == diff / diff);
  CHECK(approx_diff / approx_neg_diff == diff / neg_diff);
  CHECK(approx_diff / neg_diff == diff / neg_diff);
  CHECK(diff / approx_neg_diff == diff / neg_diff);

  const TimeDelta::rational_t third{1, 3};
  const TimeDelta::rational_t quarter{1, 4};
  CHECK(approx_diff * 3 == diff * 3);
  CHECK(approx_diff * quarter == diff * quarter);
  CHECK(3 * approx_diff == 3 * diff);
  CHECK(quarter * approx_diff == quarter * diff);
  CHECK(approx_diff / third == diff / third);
  CHECK(approx_diff / 4 == diff / 4);

  CHECK(get_output(approx1) == get_output(approx1.value()));
  CHECK(get_output(approx_diff) == get_output(approx_diff.value()));
}
