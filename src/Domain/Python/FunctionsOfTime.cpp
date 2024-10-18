// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/FunctionsOfTime.hpp"

#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

void bind_functions_of_time(py::module& m) {  // NOLINT
  domain::FunctionsOfTime::register_derived_with_charm();
  py::class_<FunctionsOfTime::FunctionOfTime>(m, "FunctionOfTime")
      .def("time_bounds", &FunctionsOfTime::FunctionOfTime::time_bounds)
      .def("func", &FunctionsOfTime::FunctionOfTime::func, py::arg("t"))
      .def("func_and_deriv", &FunctionsOfTime::FunctionOfTime::func_and_deriv,
           py::arg("t"))
      .def("func_and_2_derivs",
           &FunctionsOfTime::FunctionOfTime::func_and_2_derivs, py::arg("t"));
  const auto bind_piecewise_polynomial = [&]<size_t Order>() {
    const std::string class_name =
        "PiecewisePolynomial" + std::to_string(Order);
    py::class_<FunctionsOfTime::PiecewisePolynomial<Order>,
               FunctionsOfTime::FunctionOfTime>(m, class_name.c_str())
        .def(py::init<double, std::array<DataVector, Order + 1>, double>(),
             py::arg("time"), py::arg("initial_func_and_derivs"),
             py::arg("expiration_time"));
  };
  bind_piecewise_polynomial.template operator()<2>();
  bind_piecewise_polynomial.template operator()<3>();
  py::class_<FunctionsOfTime::QuaternionFunctionOfTime<3>,
             FunctionsOfTime::FunctionOfTime>(m, "QuaternionFunctionOfTime")
      .def(py::init<double, std::array<DataVector, 1>,
                    std::array<DataVector, 4>, double>(),
           py::arg("time"), py::arg("initial_quat_func"),
           py::arg("initial_angle_func"), py::arg("expiration_time"))
      .def("quat_func", &FunctionsOfTime::FunctionOfTime::func, py::arg("t"))
      .def("quat_func_and_deriv",
           &FunctionsOfTime::FunctionOfTime::func_and_deriv, py::arg("t"))
      .def("quat_func_and_2_derivs",
           &FunctionsOfTime::FunctionOfTime::func_and_2_derivs, py::arg("t"));
  m.def(
      "serialize_functions_of_time",
      [](const std::unordered_map<
          std::string, const domain::FunctionsOfTime::FunctionOfTime&>&
             functions_of_time) {
        std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
            unique_functions_of_time{};
        for (const auto& key_value : functions_of_time) {
          unique_functions_of_time[key_value.first] =
              functions_of_time.at(key_value.first).get_clone();
        }
        return serialize<std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>>(
            unique_functions_of_time);
      },
      py::arg("functions_of_time"));
  m.def(
      "deserialize_functions_of_time",
      [](const std::vector<char>& serialized_functions_of_time) {
        return deserialize<std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>>(
            serialized_functions_of_time.data());
      },
      py::arg("serialized_functions_of_time"));
}
}  // namespace domain::py_bindings
