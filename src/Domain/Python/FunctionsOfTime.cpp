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
