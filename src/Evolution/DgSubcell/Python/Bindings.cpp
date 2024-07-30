// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace evolution::dg::subcell {
namespace {
template <size_t Dim>
void impl(py::module& m) {  // NOLINT
  m.def("mesh", py::overload_cast<const Mesh<Dim>&>(&fd::mesh<Dim>),
        py::arg("mesh"));
  m.def(
      "project",
      +[](const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
          const Index<Dim>& subcell_extents) {
        return fd::project(dg_u, dg_mesh, subcell_extents);
      },
      py::arg("dg_u"), py::arg("dg_mesh"), py::arg("subcell_extents"));
  m.def(
      "reconstruct",
      +[](const DataVector& subcell_u_times_projected_det_jac,
          const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
          const fd::ReconstructionMethod reconstruction_method) {
        return fd::reconstruct(subcell_u_times_projected_det_jac, dg_mesh,
                               subcell_extents, reconstruction_method);
      },
      py::arg("subcell_u_times_projected_det_jac"), py::arg("dg_mesh"),
      py::arg("subcell_extents"), py::arg("reconstruction_method"));
  const auto bind_tensor =
      [&m]<typename TensorType>(const TensorType& /*meta*/) {
        m.def("persson_tci",
              &persson_tci<Dim, typename TensorType::symmetry,
                           typename TensorType::index_list>,
              py::arg("tensor"), py::arg("dg_mesh"), py::arg("alpha"),
              py::arg("num_highest_modes"));
      };
  bind_tensor(Scalar<DataVector>{});
  bind_tensor(tnsr::i<DataVector, Dim>{});
  bind_tensor(tnsr::I<DataVector, Dim>{});
}
}  // namespace

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Domain.CoordinateMaps");
  py::enum_<fd::ReconstructionMethod>(m, "FdReconstructionMethod")
      .value("DimByDim", fd::ReconstructionMethod::DimByDim)
      .value("AllDimsAtOnce", fd::ReconstructionMethod::AllDimsAtOnce);
  impl<1>(m);
  impl<2>(m);
  impl<3>(m);
}
}  // namespace evolution::dg::subcell
