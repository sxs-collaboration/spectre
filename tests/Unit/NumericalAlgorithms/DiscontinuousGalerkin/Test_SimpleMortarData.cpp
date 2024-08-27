// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Literals.hpp"

// for __decay_and_strip<>::__type

SPECTRE_TEST_CASE("Unit.DG.SimpleMortarData", "[Unit][NumericalAlgorithms]") {
  {
    dg::SimpleMortarData<size_t, std::string, double> data;
    data = serialize_and_deserialize(data);
    data.local_insert(0, "string 1");
    data = serialize_and_deserialize(data);
    data.remote_insert(0, 1.234);
    CHECK(data.local_data(0) == "string 1");
    CHECK(data.remote_data(0) == 1.234);
    CHECK(data.extract() == std::make_pair("string 1"s, 1.234));
    data = serialize_and_deserialize(data);
    data.remote_insert(1, 2.345);
    data = serialize_and_deserialize(data);
    data.local_insert(1, "string 2");
    CHECK(data.extract() == std::make_pair("string 2"s, 2.345));
  }

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        dg::SimpleMortarData<size_t, std::string, double> data;
        data.remote_insert(0, 1.234);
        data.local_data(0);
      }()),
      Catch::Matchers::ContainsSubstring("Local data not available."));
  CHECK_THROWS_WITH(([]() {
                      dg::SimpleMortarData<size_t, std::string, double> data;
                      data.local_insert(1, "");
                      data.local_data(0);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "Only have local data at temporal_id"));
  CHECK_THROWS_WITH(
      ([]() {
        dg::SimpleMortarData<size_t, std::string, double> data;
        data.remote_insert(0, 0.);
        data.local_insert(1, "");
      }()),
      Catch::Matchers::ContainsSubstring(
          "Received local data at 1, but already have remote data at 0"));
  CHECK_THROWS_WITH(
      ([]() {
        dg::SimpleMortarData<size_t, std::string, double> data;
        data.local_insert(1, "");
        data.remote_insert(0, 0.);
      }()),
      Catch::Matchers::ContainsSubstring(
          "Received remote data at 0, but already have local data at 1"));
  CHECK_THROWS_WITH(
      ([]() {
        dg::SimpleMortarData<size_t, std::string, double> data;
        data.local_insert(1, "");
        data.local_insert(1, "");
      }()),
      Catch::Matchers::ContainsSubstring("Already received local data"));
  CHECK_THROWS_WITH(
      ([]() {
        dg::SimpleMortarData<size_t, std::string, double> data;
        data.remote_insert(0, 0.);
        data.remote_insert(0, 0.);
      }()),
      Catch::Matchers::ContainsSubstring("Already received remote data"));
  CHECK_THROWS_WITH(
      ([]() {
        dg::SimpleMortarData<size_t, std::string, double> data;
        data.extract();
      }()),
      Catch::Matchers::ContainsSubstring(
          "Tried to extract boundary data, but do not have any data"));
  CHECK_THROWS_WITH(
      ([]() {
        dg::SimpleMortarData<size_t, std::string, double> data;
        data.local_insert(1, "");
        data.extract();
      }()),
      Catch::Matchers::ContainsSubstring(
          "Tried to extract boundary data, but do not have remote data"));
  CHECK_THROWS_WITH(
      ([]() {
        dg::SimpleMortarData<size_t, std::string, double> data;
        data.remote_insert(0, 0.);
        data.extract();
      }()),
      Catch::Matchers::ContainsSubstring(
          "Tried to extract boundary data, but do not have local data"));
#endif
}
