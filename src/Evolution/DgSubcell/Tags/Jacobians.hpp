// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/ElementMap.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace evolution::dg::subcell::fd::Tags {
/// \brief The inverse Jacobian from the element logical frame to the grid frame
/// at the cell centers.
///
/// Specifically, \f$\partial x^{\bar{i}} / \partial x^i\f$, where \f$\bar{i}\f$
/// denotes the element logical frame and \f$i\f$ denotes the grid frame.
///
/// \note stored as a `std::optional` so we can reset it when switching to DG
/// and reduce the memory footprint.
template <size_t Dim>
struct InverseJacobianLogicalToGrid : db::SimpleTag {
  static std::string name() { return "InverseJacobian(Logical,Grid)"; }
  using type =
      ::InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Grid>;
};

/// \brief The determinant of the inverse Jacobian from the element logical
/// frame to the grid frame at the cell centers.
///
/// \note stored as a `std::optional` so we can reset it when switching to DG
/// and reduce the memory footprint.
struct DetInverseJacobianLogicalToGrid : db::SimpleTag {
  static std::string name() { return "Det(InverseJacobian(Logical,Grid))"; }
  using type = Scalar<DataVector>;
};

/// \brief The inverse Jacobian from the element logical frame to the inertial
/// frame at the cell centers.
///
/// Specifically, \f$\partial x^{\bar{i}} / \partial x^i\f$, where \f$\bar{i}\f$
/// denotes the element logical frame and \f$i\f$ denotes the inertial frame.
///
/// \note stored as a `std::optional` so we can reset it when switching to DG
/// and reduce the memory footprint (is it useful for a ComputeTag though?).
template <size_t Dim>
struct InverseJacobianLogicalToInertial : db::SimpleTag {
  static std::string name() { return "InverseJacobian(Logical,Inertial)"; }
  using type = ::InverseJacobian < DataVector, Dim, Frame::ElementLogical,
        Frame::Inertial >;
};

/// Compute item for the inverse jacobian matrix from logical to
/// grid coordinates
template <typename MapTagLogicalToGrid, size_t Dim>
struct InverseJacobianLogicalToGridCompute : InverseJacobianLogicalToGrid<Dim>,
                                             db::ComputeTag {
  static constexpr size_t dim = Dim;
  using base = InverseJacobianLogicalToGrid<Dim>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      MapTagLogicalToGrid,
      evolution::dg::subcell::Tags::Coordinates<dim, Frame::ElementLogical>>;
  static void function(
      const gsl::not_null<return_type*> inverse_jacobian_logical_to_grid,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const tnsr::I<DataVector, dim, Frame::ElementLogical>& logical_coords) {
    *inverse_jacobian_logical_to_grid =
        logical_to_grid_map.inv_jacobian(logical_coords);
  }
};

/// Compute item for the determinant of the inverse jacobian matrix
/// from logical to grid coordinates
template <size_t Dim>
struct DetInverseJacobianLogicalToGridCompute
    : DetInverseJacobianLogicalToGrid,
      db::ComputeTag {
  static constexpr size_t dim = Dim;
  using base = DetInverseJacobianLogicalToGrid;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<InverseJacobianLogicalToGrid<Dim>>;
  static void function(
      const gsl::not_null<return_type*> det_inverse_jacobian_logical_to_grid,
      const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                              Frame::Grid>
          inverse_jacobian_logical_to_grid) {
    *det_inverse_jacobian_logical_to_grid =
        determinant(inverse_jacobian_logical_to_grid);
  }
};

/// Compute item for the inverse jacobian matrix from logical to
/// inertial coordinates
template <typename MapTagGridToInertial, size_t Dim>
struct InverseJacobianLogicalToInertialCompute
    : InverseJacobianLogicalToInertial<Dim>,
      db::ComputeTag {
  static constexpr size_t dim = Dim;
  using base = InverseJacobianLogicalToInertial<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<MapTagGridToInertial,
                 evolution::dg::subcell::Tags::Coordinates<dim, Frame::Grid>,
                 ::Tags::Time, ::domain::Tags::FunctionsOfTime,
                 InverseJacobianLogicalToGrid<Dim>>;
  static void function(
      const gsl::not_null<return_type*> inverse_jacobian_logical_to_inertial,
      const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, dim>&
          grid_to_inertial_map,
      const tnsr::I<DataVector, dim, Frame::Grid>& grid_coords,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const ::InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                              Frame::Grid>& inverse_jacobian_logical_to_grid) {
    if (grid_to_inertial_map.is_identity()) {
      // Optimization for time-independent maps; we just point to the
      // logical-to-grid inverse jacobian.
      const size_t num_pts = inverse_jacobian_logical_to_grid[0].size();
      for (size_t storage_index = 0;
           storage_index < inverse_jacobian_logical_to_grid.size();
           ++storage_index) {
        make_const_view(
            make_not_null(&std::as_const(
                (*inverse_jacobian_logical_to_inertial)[storage_index])),
            inverse_jacobian_logical_to_grid[storage_index], 0, num_pts);
      }
    } else {
      set_number_of_grid_points(inverse_jacobian_logical_to_inertial,
                                get<0>(grid_coords).size());
      // Get grid to inertial inverse jacobian
      const auto& inverse_jacobian_grid_to_inertial =
          grid_to_inertial_map.inv_jacobian(grid_coords, time,
                                            functions_of_time);
      for (size_t i = 0; i < Dim; i++) {
        for (size_t j = 0; j < Dim; j++) {
          auto& inv_jacobian_component =
              inverse_jacobian_logical_to_inertial->get(i, j);
          inv_jacobian_component = 0.;
          for (size_t k = 0; k < Dim; k++) {
            inv_jacobian_component +=
                inverse_jacobian_logical_to_grid.get(i, k) *
                inverse_jacobian_grid_to_inertial.get(k, j);
          }
        }
      }
    }
  }
};

}  // namespace evolution::dg::subcell::fd::Tags
