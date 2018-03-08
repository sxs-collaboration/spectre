// Distributed under the MIT License.
// See LICENSE.txt for detai

#include <array>
#include <string>
#include <vector>

#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestingFramework.hpp"

SPECTRE_TEST_CASE("Unit.Pypp.none", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  pypp::call<pypp::None>("PyppPyTests", "test_none");
  CHECK_THROWS(pypp::call<pypp::None>("PyppPyTests", "test_numeric", 1, 2));
  CHECK_THROWS(pypp::call<std::string>("PyppPyTests", "test_none"));
}

SPECTRE_TEST_CASE("Unit.Pypp.std::string", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  const auto ret = pypp::call<std::string>("PyppPyTests", "test_string",
                                           std::string("test string"));
  CHECK(ret == std::string("back test string"));
  CHECK_THROWS(pypp::call<std::string>("PyppPyTests", "test_string",
                                       std::string("test string_")));
  CHECK_THROWS(pypp::call<double>("PyppPyTests", "test_string",
                                  std::string("test string")));
}

SPECTRE_TEST_CASE("Unit.Pypp.int", "[Pypp][Unit]") {
  /// [pypp_int_test]
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  const auto ret = pypp::call<long>("PyppPyTests", "test_numeric", 3, 4);
  CHECK(ret == 3 * 4);
  /// [pypp_int_test]
  CHECK_THROWS(pypp::call<double>("PyppPyTests", "test_numeric", 3, 4));
  CHECK_THROWS(pypp::call<void*>("PyppPyTests", "test_numeric", 3, 4));
  CHECK_THROWS(pypp::call<long>("PyppPyTests", "test_none"));
}

SPECTRE_TEST_CASE("Unit.Pypp.long", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  const auto ret = pypp::call<long>("PyppPyTests", "test_numeric", 3l, 4l);
  CHECK(ret == 3l * 4l);
  CHECK_THROWS(pypp::call<double>("PyppPyTests", "test_numeric", 3l, 4l));
  CHECK_THROWS(pypp::call<long>("PyppPyTests", "test_numeric", 3.0, 3.74));
}

SPECTRE_TEST_CASE("Unit.Pypp.unsigned_long", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  const auto ret =
      pypp::call<unsigned long>("PyppPyTests", "test_numeric", 3ul, 4ul);
  CHECK(ret == 3ul * 4ul);
  CHECK_THROWS(pypp::call<double>("PyppPyTests", "test_numeric", 3ul, 4ul));
  CHECK_THROWS(
      pypp::call<unsigned long>("PyppPyTests", "test_numeric", 3.0, 3.74));
}

SPECTRE_TEST_CASE("Unit.Pypp.double", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  const auto ret =
      pypp::call<double>("PyppPyTests", "test_numeric", 3.49582, 3);
  CHECK(ret == 3.0 * 3.49582);
  CHECK_THROWS(pypp::call<long>("PyppPyTests", "test_numeric", 3.8, 3.9));
  CHECK_THROWS(pypp::call<double>("PyppPyTests", "test_numeric", 3ul, 3ul));
}

SPECTRE_TEST_CASE("Unit.Pypp.std::vector", "[Pypp][Unit]") {
  /// [pypp_vector_test]
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  const auto ret = pypp::call<std::vector<double>>(
      "PyppPyTests", "test_vector", std::vector<double>{1.3, 4.9},
      std::vector<double>{4.2, 6.8});
  CHECK(approx(ret[0]) == 1.3 * 4.2);
  CHECK(approx(ret[1]) == 4.9 * 6.8);
  /// [pypp_vector_test]
  CHECK_THROWS(pypp::call<std::string>("PyppPyTests", "test_vector",
                                       std::vector<double>{1.3, 4.9},
                                       std::vector<double>{4.2, 6.8}));
}

SPECTRE_TEST_CASE("Unit.Pypp.std::array", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  // std::arrays and std::vectors should both convert to lists in python so this
  // test calls the same python function as the vector test
  const auto ret = pypp::call<std::array<double, 2>>(
      "PyppPyTests", "test_vector", std::array<double, 2>{{1.3, 4.9}},
      std::array<double, 2>{{4.2, 6.8}});
  CHECK(approx(ret[0]) == 1.3 * 4.2);
  CHECK(approx(ret[1]) == 4.9 * 6.8);
  CHECK_THROWS(pypp::call<double>("PyppPyTests", "test_vector",
                                  std::array<double, 2>{{1.3, 4.9}},
                                  std::array<double, 2>{{4.2, 6.8}}));
}

SPECTRE_TEST_CASE("Unit.Pypp.DataVector", "[Pypp][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Pypp/"};
  const auto ret = pypp::call<DataVector>(
      "numpy", "multiply", DataVector{1.3, 4.9}, DataVector{4.2, 6.8});
  CHECK(approx(ret[0]) == 1.3 * 4.2);
  CHECK(approx(ret[1]) == 4.9 * 6.8);
  CHECK_THROWS(pypp::call<std::string>(
      "numpy", "multiply", DataVector{1.3, 4.9}, DataVector{4.2, 6.8}));
  CHECK_THROWS(pypp::call<DataVector>("PyppPyTests", "two_dim_ndarray"));
  CHECK_THROWS(pypp::call<DataVector>("PyppPyTests", "ndarray_of_floats"));
}
