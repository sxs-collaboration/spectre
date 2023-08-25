// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Events::Tags {
/// \brief The mesh for the observation computational grid. For hybrid methods
/// like DG-FD the observer mesh changes throughout the evolution.
template <size_t Dim>
struct ObserverMesh : db::SimpleTag {
  using type = ::Mesh<Dim>;
};

/// \brief Sets the `ObserverMesh` to `domain::Tags::Mesh`
///
/// This is what you would use for a single numerical method simulation. Hybrid
/// methods will supply their own tags.
template <size_t Dim>
struct ObserverMeshCompute : ObserverMesh<Dim>, db::ComputeTag {
  using base = ObserverMesh<Dim>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<::domain::Tags::Mesh<Dim>>;
  static void function(const gsl::not_null<return_type*> observer_mesh,
                       const ::Mesh<Dim>& mesh) {
    *observer_mesh = mesh;
  }
};

/*!
 * \brief The coordinates used for observation.
 *
 * In methods like DG-FD the mesh and coordinates change throughout the
 * simulation, so we need to always grab the right ones.
 */
template <size_t Dim, typename Fr>
struct ObserverCoordinates : db::SimpleTag {
  static std::string name() { return get_output(Fr{}) + "Coordinates"; }
  using type = tnsr::I<DataVector, Dim, Fr>;
};

/// \brief Sets the `ObserverCoordinates` to `domain::Tags::Coordinates`
///
/// This is what you would use for a single numerical method simulation. Hybrid
/// methods will supply their own tags.
template <size_t Dim, typename Fr>
struct ObserverCoordinatesCompute : ObserverCoordinates<Dim, Fr>,
                                    db::ComputeTag {
  using base = ObserverCoordinates<Dim, Fr>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<::domain::Tags::Coordinates<Dim, Fr>>;
  static void function(const gsl::not_null<return_type*> observer_coords,
                       const return_type& coords) {
    for (size_t i = 0; i < Dim; ++i) {
      observer_coords->get(i).set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          make_not_null(&const_cast<DataVector&>(coords.get(i))));
    }
  }
};

/*!
 * \brief The inverse Jacobian used for observation.
 *
 * In methods like DG-FD the mesh and inverse Jacobian change throughout the
 * simulation, so we need to always grab the right ones.
 */
template <size_t Dim, typename SourceFrame, typename TargetFrame>
struct ObserverInverseJacobian : db::SimpleTag {
  static std::string name() {
    return "InverseJacobian(" + get_output(SourceFrame{}) + "," +
           get_output(TargetFrame{}) + ")";
  }
  using type = ::InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>;
};

/// \brief Sets the `ObserverInverseJacobian` to `domain::Tags::InverseJacobian`
///
/// This is what you would use for a single numerical method simulation. Hybrid
/// methods will supply their own tags.
template <size_t Dim, typename SourceFrame, typename TargetFrame>
struct ObserverInverseJacobianCompute
    : ObserverInverseJacobian<Dim, SourceFrame, TargetFrame>,
      db::ComputeTag {
  using base = ObserverInverseJacobian<Dim, SourceFrame, TargetFrame>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      ::domain::Tags::InverseJacobian<Dim, SourceFrame, TargetFrame>>;
  static void function(
      const gsl::not_null<return_type*> observer_inverse_jacobian,
      const return_type& inverse_jacobian) {
    for (size_t i = 0; i < inverse_jacobian.size(); ++i) {
      (*observer_inverse_jacobian)[i].set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          make_not_null(&const_cast<DataVector&>(inverse_jacobian[i])));
    }
  }
};

/*!
 * \brief The Jacobian used for observation.
 *
 * In methods like DG-FD the mesh and  Jacobian change throughout the
 * simulation, so we need to always grab the right ones.
 */
template <size_t Dim, typename SourceFrame, typename TargetFrame>
struct ObserverJacobian : db::SimpleTag {
  static std::string name() {
    return "Jacobian(" + get_output(SourceFrame{}) + "," +
           get_output(TargetFrame{}) + ")";
  }
  using type = ::Jacobian<DataVector, Dim, SourceFrame, TargetFrame>;
};

/// \brief Sets the `ObserverJacobian` to `domain::Tags::Jacobian`
///
/// This is what you would use for a single numerical method simulation. Hybrid
/// methods will supply their own tags.
template <size_t Dim, typename SourceFrame, typename TargetFrame>
struct ObserverJacobianCompute
    : ObserverJacobian<Dim, SourceFrame, TargetFrame>,
      db::ComputeTag {
  using base = ObserverJacobian<Dim, SourceFrame, TargetFrame>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::domain::Tags::Jacobian<Dim, SourceFrame, TargetFrame>>;
  static void function(const gsl::not_null<return_type*> observer_jacobian,
                       const return_type& jacobian) {
    for (size_t i = 0; i < jacobian.size(); ++i) {
      (*observer_jacobian)[i].set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          make_not_null(&const_cast<DataVector&>(jacobian[i])));
    }
  }
};

/*!
 * \brief The determinant of the inverse Jacobian used for observation.
 *
 * In methods like DG-FD the mesh and inverse Jacobian change throughout the
 * simulation, so we need to always grab the right ones.
 */
template <typename SourceFrame, typename TargetFrame>
struct ObserverDetInvJacobian : db::SimpleTag {
  static std::string name() {
    return "DetInvJacobian(" + get_output(SourceFrame{}) + "," +
           get_output(TargetFrame{}) + ")";
  }
  using type = Scalar<DataVector>;
};

/// \brief Sets the `ObserverDetInvJacobian` to `domain::Tags::DetInvJacobian`
///
/// This is what you would use for a single numerical method simulation. Hybrid
/// methods will supply their own tags.
template <typename SourceFrame, typename TargetFrame>
struct ObserverDetInvJacobianCompute
    : ObserverDetInvJacobian<SourceFrame, TargetFrame>,
      db::ComputeTag {
  using base = ObserverDetInvJacobian<SourceFrame, TargetFrame>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::domain::Tags::DetInvJacobian<SourceFrame, TargetFrame>>;
  static void function(
      const gsl::not_null<return_type*> observer_det_inverse_jacobian,
      const return_type& det_inverse_jacobian) {
    for (size_t i = 0; i < det_inverse_jacobian.size(); ++i) {
      (*observer_det_inverse_jacobian)[i].set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          make_not_null(&const_cast<DataVector&>(det_inverse_jacobian[i])));
    }
  }
};

/// \brief The mesh velocity used for observations
///
/// The type is a `std::optional`, which when it is not set indicates that the
/// mesh is not moving.
template <size_t Dim, typename Frame = ::Frame::Inertial>
struct ObserverMeshVelocity : db::SimpleTag {
  static std::string name() { return get_output(Frame{}) + "MeshVelocity"; }
  using type = std::optional<tnsr::I<DataVector, Dim, Frame>>;
};

/// \brief Sets the `ObserverMeshVelocty` to `domain::Tags::MeshVelocty`
///
/// This is what you would use for a single numerical method simulation. Hybrid
/// methods will supply their own tags.
template <size_t Dim, typename Frame = ::Frame::Inertial>
struct ObserverMeshVelocityCompute : ObserverMeshVelocity<Dim, Frame>,
                                     db::ComputeTag {
  using base = ObserverMeshVelocity<Dim, Frame>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<::domain::Tags::MeshVelocity<Dim, Frame>>;
  static void function(const gsl::not_null<return_type*> observer_mesh_velocity,
                       const return_type& mesh_velocity) {
    if (mesh_velocity.has_value()) {
      *observer_mesh_velocity = tnsr::I<DataVector, Dim, Frame>{};
      for (size_t i = 0; i < mesh_velocity->size(); ++i) {
        observer_mesh_velocity->value()[i].set_data_ref(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            make_not_null(&const_cast<DataVector&>(mesh_velocity.value()[i])));
      }
    } else {
      *observer_mesh_velocity = std::nullopt;
    }
  }
};
}  // namespace Events::Tags
