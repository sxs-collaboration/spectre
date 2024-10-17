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
  py::class_<FunctionsOfTime::FunctionOfTime,
             std::shared_ptr<FunctionsOfTime::FunctionOfTime>>(m,
                                                               "FunctionOfTime")
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
               FunctionsOfTime::FunctionOfTime,
               std::shared_ptr<FunctionsOfTime::PiecewisePolynomial<Order>>>(
        m, class_name.c_str())
        .def(py::init(
            [](double time,
               std::array<DataVector, Order + 1>& initial_func_and_derivs,
               double expiration_time) {
              return FunctionsOfTime::PiecewisePolynomial<Order>(
                  time, initial_func_and_derivs, expiration_time);
            }))
        .def("time_bounds", &FunctionsOfTime::FunctionOfTime::time_bounds)
        .def("func", &FunctionsOfTime::FunctionOfTime::func, py::arg("t"))
        .def("func_and_deriv", &FunctionsOfTime::FunctionOfTime::func_and_deriv,
             py::arg("t"))
        .def("func_and_2_derivs",
             &FunctionsOfTime::FunctionOfTime::func_and_2_derivs, py::arg("t"));
  };
  bind_piecewise_polynomial.template operator()<2>();
  bind_piecewise_polynomial.template operator()<3>();
  py::class_<FunctionsOfTime::QuaternionFunctionOfTime<3>,
             FunctionsOfTime::FunctionOfTime,
             std::shared_ptr<FunctionsOfTime::QuaternionFunctionOfTime<3>>>(
      m, "QuaternionFunctionOfTime")
      .def(
          py::init([](double time, std::array<DataVector, 1>& initial_quat_func,
                      std::array<DataVector, 4>& initial_angle_func,
                      double expiration_time) {
            return FunctionsOfTime::QuaternionFunctionOfTime<3>(
                time, initial_quat_func, initial_angle_func, expiration_time);
          }))
      .def("time_bounds", &FunctionsOfTime::FunctionOfTime::time_bounds)
      .def("func", &FunctionsOfTime::FunctionOfTime::func, py::arg("t"))
      .def("func_and_deriv", &FunctionsOfTime::FunctionOfTime::func_and_deriv,
           py::arg("t"))
      .def("func_and_2_derivs",
           &FunctionsOfTime::FunctionOfTime::func_and_2_derivs, py::arg("t"))
      .def("quat_func", &FunctionsOfTime::FunctionOfTime::func, py::arg("t"))
      .def("quat_func_and_deriv",
           &FunctionsOfTime::FunctionOfTime::func_and_deriv, py::arg("t"))
      .def("quat_func_and_2_derivs",
           &FunctionsOfTime::FunctionOfTime::func_and_2_derivs, py::arg("t"));
  m.def(
      "serialize_functions_of_time",
      [](const std::unordered_map<
          std::string,
          std::shared_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
             shared_functions_of_time) {
        std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
            unique_functions_of_time{};
        for (const auto& key_value : shared_functions_of_time) {
          unique_functions_of_time[key_value.first] =
              shared_functions_of_time.at(key_value.first)->get_clone();
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
            std::shared_ptr<domain::FunctionsOfTime::FunctionOfTime>>>(
            serialized_functions_of_time.data());
      },
      py::arg("serialized_functions_of_time"));
}
}  // namespace domain::py_bindings
