// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace py = pybind11;

namespace gr::surfaces::py_bindings {

template <typename Frame>
void bind_horizon_quantities_impl(py::module& m) {
  static constexpr size_t Dim = 3;
  m.def(
      "horizon_quantities",
      [](ylm::Strahlkorper<Frame> horizon,
         tnsr::ii<DataVector, Dim, Frame> spatial_metric,
         tnsr::II<DataVector, Dim, Frame> inv_spatial_metric,
         tnsr::ii<DataVector, Dim, Frame> extrinsic_curvature,
         tnsr::Ijj<DataVector, Dim, Frame> spatial_christoffel_second_kind,
         tnsr::ii<DataVector, Dim, Frame> spatial_ricci) -> py::dict {
        const auto box = db::create<
            tmpl::append<tmpl::list<ylm::Tags::Strahlkorper<Frame>>,
                         ::ah::vars_to_interpolate_to_target<Dim, Frame>>,
            tmpl::push_back<::ah::compute_items_on_target<Dim, Frame>,
                            gr::surfaces::Tags::DimensionlessSpinVectorCompute<
                                Frame, Frame>>>(
            std::move(horizon), std::move(spatial_metric),
            std::move(inv_spatial_metric), std::move(extrinsic_curvature),
            std::move(spatial_christoffel_second_kind),
            std::move(spatial_ricci));
        py::dict result{};
        result["Area"] = db::get<gr::surfaces::Tags::Area>(box);
        result["IrreducibleMass"] =
            db::get<gr::surfaces::Tags::IrreducibleMass>(box);
        result["MaxRicciScalar"] = db::get<ylm::Tags::MaxRicciScalar>(box);
        result["MinRicciScalar"] = db::get<ylm::Tags::MinRicciScalar>(box);
        result["ChristodoulouMass"] =
            db::get<gr::surfaces::Tags::ChristodoulouMass>(box);
        result["DimensionlessSpinMagnitude"] =
            db::get<gr::surfaces::Tags::DimensionlessSpinMagnitude<Frame>>(box);
        result["DimensionlessSpinVector"] =
            db::get<gr::surfaces::Tags::DimensionlessSpinVector<Frame>>(box);
        return result;
      },
      py::arg("horizon"), py::arg("spatial_metric"),
      py::arg("inv_spatial_metric"), py::arg("extrinsic_curvature"),
      py::arg("spatial_christoffel_second_kind"), py::arg("spatial_ricci"));
}

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.SphericalHarmonics");
  bind_horizon_quantities_impl<Frame::Inertial>(m);
}

}  // namespace gr::surfaces::py_bindings
