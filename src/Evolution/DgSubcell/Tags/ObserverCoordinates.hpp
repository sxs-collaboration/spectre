// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::Tags {
/*!
 * \brief "Computes" the active coordinates by setting the `DataVector`s to
 * point into the coordinates of either the DG or subcell grid.
 */
template <size_t Dim, typename Fr>
struct ObserverCoordinatesCompute
    : db::ComputeTag,
      ::Events::Tags::ObserverCoordinates<Dim, Fr> {
  using base = ::Events::Tags::ObserverCoordinates<Dim, Fr>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<ActiveGrid, Coordinates<Dim, Fr>,
                                   ::domain::Tags::Coordinates<Dim, Fr>>;
  static void function(const gsl::not_null<return_type*> active_coords,
                       const subcell::ActiveGrid active_grid,
                       const tnsr::I<DataVector, Dim, Fr>& subcell_coords,
                       const tnsr::I<DataVector, Dim, Fr>& dg_coords) {
    const auto set_to_refs =
        [&active_coords](const tnsr::I<DataVector, Dim, Fr>& coords) {
          for (size_t i = 0; i < Dim; ++i) {
            active_coords->get(i).set_data_ref(
                make_not_null(&const_cast<DataVector&>(coords.get(i))));
          }
        };
    if (active_grid == subcell::ActiveGrid::Dg) {
      set_to_refs(dg_coords);
    } else {
      ASSERT(active_grid == subcell::ActiveGrid::Subcell,
             "ActiveGrid should be subcell if it isn't DG. Maybe an extra enum "
             "entry was added?");
      set_to_refs(subcell_coords);
    }
  }
};

/*!
 * \brief Computes the active inverse Jacobian.
 */
template <size_t Dim, typename SourceFrame, typename TargetFrame>
struct ObserverInverseJacobianCompute
    : ::Events::Tags::ObserverInverseJacobian<Dim, SourceFrame, TargetFrame>,
      db::ComputeTag {
  using base =
      ::Events::Tags::ObserverInverseJacobian<Dim, SourceFrame, TargetFrame>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Events::Tags::ObserverCoordinates<Dim, SourceFrame>,
                 ::Tags::Time, domain::Tags::FunctionsOfTime,
                 domain::Tags::ElementMap<Dim, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>>;
  static void function(
      const gsl::not_null<return_type*> observer_inverse_jacobian,
      const tnsr::I<DataVector, Dim, SourceFrame>& source_coords,
      [[maybe_unused]] const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
          grid_to_inertial_map) {
    static_assert(std::is_same_v<SourceFrame, Frame::ElementLogical> or
                  std::is_same_v<SourceFrame, Frame::Grid>);
    static_assert(std::is_same_v<TargetFrame, Frame::Inertial> or
                  std::is_same_v<TargetFrame, Frame::Grid>);
    static_assert(not std::is_same_v<SourceFrame, TargetFrame>);
    if constexpr (std::is_same_v<SourceFrame, Frame::ElementLogical>) {
      if constexpr (std::is_same_v<TargetFrame, Frame::Inertial>) {
        if (grid_to_inertial_map.is_identity()) {
          auto logical_to_grid_inv_jac =
              logical_to_grid_map.inv_jacobian(source_coords);
          for (size_t i = 0; i < observer_inverse_jacobian->size(); ++i) {
            (*observer_inverse_jacobian)[i] =
                std::move(logical_to_grid_inv_jac[i]);
          }
        } else {
          const auto logical_to_grid_inv_jac =
              logical_to_grid_map.inv_jacobian(source_coords);
          const auto grid_to_inertial_inv_jac =
              grid_to_inertial_map.inv_jacobian(
                  logical_to_grid_map(source_coords), time, functions_of_time);
          for (size_t logical_i = 0; logical_i < Dim; ++logical_i) {
            for (size_t inertial_i = 0; inertial_i < Dim; ++inertial_i) {
              observer_inverse_jacobian->get(logical_i, inertial_i) =
                  logical_to_grid_inv_jac.get(logical_i, 0) *
                  grid_to_inertial_inv_jac.get(0, inertial_i);
              for (size_t grid_i = 1; grid_i < Dim; ++grid_i) {
                observer_inverse_jacobian->get(logical_i, inertial_i) +=
                    logical_to_grid_inv_jac.get(logical_i, grid_i) *
                    grid_to_inertial_inv_jac.get(grid_i, inertial_i);
              }
            }
          }
        }
      } else {
        *observer_inverse_jacobian =
            logical_to_grid_map.inv_jacobian(source_coords);
      }
    } else {
      *observer_inverse_jacobian = grid_to_inertial_map.inv_jacobian(
          source_coords, time, functions_of_time);
    }
  }
};

/*!
 * \brief Computes the active Jacobian and determinant of the inverse Jacobian.
 */
template <size_t Dim, typename SourceFrame, typename TargetFrame>
struct ObserverJacobianAndDetInvJacobian
    : ::Tags::Variables<tmpl::list<
          ::Events::Tags::ObserverDetInvJacobian<SourceFrame, TargetFrame>,
          ::Events::Tags::ObserverJacobian<Dim, SourceFrame, TargetFrame>>>,
      db::ComputeTag {
  using base = ::Tags::Variables<tmpl::list<
      ::Events::Tags::ObserverDetInvJacobian<SourceFrame, TargetFrame>,
      ::Events::Tags::ObserverJacobian<Dim, SourceFrame, TargetFrame>>>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      ::Events::Tags::ObserverInverseJacobian<Dim, SourceFrame, TargetFrame>>;
  static void function(const gsl::not_null<return_type*> result,
                       const InverseJacobian<DataVector, Dim, SourceFrame,
                                             TargetFrame>& inv_jacobian) {
    determinant_and_inverse(result, inv_jacobian);
  }
};
}  // namespace evolution::dg::subcell::Tags
