// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/External/InterpolateFromFuka.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.IO.External.InterpolateFromFuka", "[Unit][IO]") {
  std::mutex fuka_lock{};
  // Get example data directory from environment variable. The example ID is
  // in `FUKA_ROOT/codes/PythonTools/Example_id` unless installed elsewhere.
  const char* example_id_dir_ptr = std::getenv("FUKA_EXAMPLE_ID_DIR");
  REQUIRE(example_id_dir_ptr != nullptr);
  const std::string example_id_dir{example_id_dir_ptr};
  REQUIRE_FALSE(example_id_dir.empty());
  CAPTURE(example_id_dir);
  {
    INFO("BH");
    const tnsr::I<DataVector, 3> coords{{{{2.0}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Bh>(
        make_not_null(&fuka_lock),
        example_id_dir + "/converged_BH_TOTAL_BC.0.5.0.0.09.info", coords);
    CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataVector>>(fuka_data)),
                          DataVector{0.78166130712794868});
  }
  {
    INFO("BBH");
    const tnsr::I<DataVector, 3> coords{{{{0.0}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Bbh>(
        make_not_null(&fuka_lock),
        example_id_dir + "/converged_BBH_TOTAL_BC.10.0.0.1.q1.0.0.09.info",
        coords);
    CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataVector>>(fuka_data)),
                          DataVector{0.82006289882662431});
  }
  {
    INFO("NS");
    const tnsr::I<DataVector, 3> coords{{{{0.0}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Ns>(
        make_not_null(&fuka_lock),
        example_id_dir + "/converged_NS_TOTAL_BC.togashi.2.23.-0.4.0.11.info",
        coords);
    CHECK_ITERABLE_APPROX(
        get(get<hydro::Tags::RestMassDensity<DataVector>>(fuka_data)),
        DataVector{0.00404310450359371});
  }
  {
    INFO("BNS");
    const tnsr::I<DataVector, 3> coords{{{{15.3}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Bns>(
        make_not_null(&fuka_lock),
        example_id_dir +
            "/converged_BNS_TOTAL.togashi.30.6.0.0.2.8.q1.0.0.09.info",
        coords);
    CHECK_ITERABLE_APPROX(
        get(get<hydro::Tags::RestMassDensity<DataVector>>(fuka_data)),
        DataVector{0.00137492312500218});
  }
  {
    INFO("BHNS");
    const tnsr::I<DataVector, 3> coords{{{{0.0}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Bhns>(
        make_not_null(&fuka_lock),
        example_id_dir +
            "/converged_BHNS_ECC_RED.togashi.35.0.6.0.52.3.6.q0.487603.0.1.11."
            "info",
        coords);
    CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataVector>>(fuka_data)),
                          DataVector{0.77494679614415585});
  }
}
