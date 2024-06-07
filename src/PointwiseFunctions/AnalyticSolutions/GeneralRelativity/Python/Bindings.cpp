// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <exception>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/TMPL.hpp"

namespace py = pybind11;

namespace gr::Solutions::py_bindings {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");

  py::class_<KerrSchild>(m, "KerrSchild")
      .def(py::init<double, const std::array<double, 3>&,
                    const std::array<double, 3>&>(),
           py::arg("mass"), py::arg("dimensionless_spin"),
           py::arg("center") = std::array<double, 3>{{0.0, 0.0, 0.0}})
      .def_property_readonly("mass", &KerrSchild::mass)
      .def_property_readonly("dimensionless_spin",
                             &KerrSchild::dimensionless_spin)
      .def_property_readonly("center", &KerrSchild::center)
      .def(
          "variables",
          [](const KerrSchild& solution, const tnsr::I<DataVector, 3>& x,
             const std::vector<std::string>& requested_quantities) -> py::dict {
            KerrSchild::IntermediateVars<DataVector, Frame::Inertial> cache(
                x.begin()->size());
            KerrSchild::IntermediateComputer<DataVector, Frame::Inertial>
                computer(solution, x);
            using available_tags =
                KerrSchild::tags<DataVector, Frame::Inertial>;
            py::dict result{};
            for (const auto& requested_quantity : requested_quantities) {
              bool found = false;
              tmpl::for_each<available_tags>([&requested_quantity, &cache,
                                              &computer, &result,
                                              &found](const auto tag_v) {
                if (found) {
                  return;
                }
                using tag = tmpl::type_from<decltype(tag_v)>;
                if (requested_quantity == db::tag_name<tag>()) {
                  result[requested_quantity.c_str()] =
                      cache.get_var(computer, tag{});
                  found = true;
                }
              });
              if (not found) {
                std::string available_quantities{};
                tmpl::for_each<available_tags>(
                    [&available_quantities](const auto tag_v) {
                      using tag = tmpl::type_from<decltype(tag_v)>;
                      available_quantities += db::tag_name<tag>() + "\n";
                    });
                throw std::invalid_argument(
                    "Requested quantity '" + requested_quantity +
                    "' is not available. "
                    // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
                    "Available quantities are:\n" +
                    available_quantities);
              }
            }
            return result;
          },
          py::arg("x"), py::arg("requested_quantities"));
}

}  // namespace gr::Solutions::py_bindings
