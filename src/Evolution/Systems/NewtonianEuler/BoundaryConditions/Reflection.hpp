// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/SliceVariable.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::BoundaryConditions {
/*!
 * \brief Reflecting boundary conditions for Newtonian hydrodynamics.
 *
 * Ghost (exterior) data 'mirrors' interior volume data with respect to the
 * boundary interface. i.e. reverses the normal component of velocity while
 * using same values for other scalar quantities.
 *
 * In the frame instantaneously moving with the same velocity as face mesh, the
 * reflection condition reads
 *
 * \f{align*}
 * \vec{u}_\text{ghost} = \vec{u}_\text{int} - 2 (\vec{u}_\text{int} \cdot
 * \hat{n}) \hat{n}
 * \f}
 *
 * where \f$\vec{u}\f$ is the fluid velocity in the moving frame, "int" stands
 * for interior, and \f$\hat{n}\f$ is the outward normal vector on the boundary
 * interface.
 *
 * Substituting \f$\vec{u} = \vec{v} - \vec{v}_m\f$, we get
 *
 * \f{align*}
 * v_\text{ghost}^i &= v_\text{int}^i - 2[(v_\text{int}^j-v_m^j)n_j]n^i
 * \f}
 *
 * where \f$v\f$ is the fluid velocity and \f$v_m\f$ is face mesh velocity.
 *
 */
template <size_t Dim>
class Reflection final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Reflecting boundary conditions for Newtonian hydrodynamics."};

  Reflection() = default;
  Reflection(Reflection&&) = default;
  Reflection& operator=(Reflection&&) = default;
  Reflection(const Reflection&) = default;
  Reflection& operator=(const Reflection&) = default;
  ~Reflection() override = default;

  explicit Reflection(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Reflection);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<Tags::MassDensity<DataVector>, Tags::Velocity<DataVector, Dim>,
                 Tags::SpecificInternalEnergy<DataVector>,
                 Tags::Pressure<DataVector>>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          momentum_density,
      const gsl::not_null<Scalar<DataVector>*> energy_density,

      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_mass_density,
      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_momentum_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_energy_density,

      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      const Scalar<DataVector>& interior_mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_specific_internal_energy,
      const Scalar<DataVector>& interior_pressure) const;

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<Dim>,
                 domain::Tags::MeshVelocity<Dim, Frame::Inertial>>;
  using fd_interior_primitive_variables_tags =
      tmpl::list<Tags::MassDensity<DataVector>, Tags::Velocity<DataVector, Dim>,
                 Tags::Pressure<DataVector>>;
  using fd_gridless_tags =
      tmpl::list<fd::Tags::Reconstructor<Dim>, ::Tags::Time,
                 domain::Tags::FunctionsOfTime,
                 domain::Tags::ElementMap<Dim, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                             Frame::Inertial>,
                 Parallel::Tags::Metavariables>;

  template <typename Metavariables>
  void fd_ghost(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const Direction<Dim>& direction,
      // interior temporary tags
      const Mesh<Dim> subcell_mesh,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
          volume_mesh_velocity,
      // interior primitive variables tags
      const Scalar<DataVector>& interior_mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_pressure,
      // gridless tags
      const fd::Reconstructor<Dim>& reconstructor, const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
          grid_to_inertial_map,
      const Metavariables& /*meta*/) const {
    const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

    const auto subcell_extents = subcell_mesh.extents();

    Variables<fd_interior_primitive_variables_tags> volume_prim_vars{};
    volume_prim_vars.initialize(subcell_extents.product());

    using MassDensity = Tags::MassDensity<DataVector>;
    using Velocity = Tags::Velocity<DataVector, Dim>;
    using Pressure = Tags::Pressure<DataVector>;
    get<MassDensity>(volume_prim_vars) = interior_mass_density;
    get<Velocity>(volume_prim_vars) = interior_velocity;
    get<Pressure>(volume_prim_vars) = interior_pressure;

    Variables<fd_interior_primitive_variables_tags> sliced_prim_vars{};
    const size_t num_sliced_pts{
        subcell_extents.slice_away(direction.dimension()).product() *
        ghost_zone_size};
    sliced_prim_vars.initialize(num_sliced_pts);

    // Slice volume primitive variables upto the depth `ghost_zone_size`
    evolution::dg::subcell::slice_variable(make_not_null(&sliced_prim_vars),
                                           volume_prim_vars, subcell_extents,
                                           ghost_zone_size, direction);

    // Construct an ourward-pointing normal vector field on the sliced subcell
    // mesh, which is needed for computing normal components of velocity at each
    // grid points.
    //
    // Note) It is assumed here that the outward normal vector field on ghost
    // points is aligned with internal grid lines and same as the outward normal
    // vector field at internal grid points. We will need a different strategy
    // when using unstructured meshes in the future.
    //
    Index<Dim> sliced_extents = subcell_extents;
    sliced_extents[direction.dimension()] = ghost_zone_size;
    const Mesh<Dim> sliced_mesh{sliced_extents.indices(),
                                Spectral::Basis::FiniteDifference,
                                Spectral::Quadrature::CellCentered};

    DirectionMap<Dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<Dim>>>>>
        normal_covectors_and_magnitudes{};
    {
      auto sliced_logical_coords = logical_coordinates(sliced_mesh);
      std::unordered_map<Direction<Dim>,
                         tnsr::i<DataVector, Dim, Frame::Inertial>>
          unnormalized_normal_covectors{};
      tnsr::i<DataVector, Dim, Frame::Inertial> unnormalized_covector{};
      for (size_t i = 0; i < Dim; ++i) {
        unnormalized_covector.get(i) =
            grid_to_inertial_map
                .inv_jacobian(logical_to_grid_map(sliced_logical_coords), time,
                              functions_of_time)
                .get(direction.dimension(), i);
      }
      unnormalized_normal_covectors[direction] = unnormalized_covector;
      Variables<tmpl::list<
          evolution::dg::Actions::detail::NormalVector<Dim>,
          evolution::dg::Actions::detail::OneOverNormalVectorMagnitude>>
          fields_on_face{sliced_mesh.number_of_grid_points()};
      normal_covectors_and_magnitudes[direction] = std::nullopt;
      using system = typename std::decay_t<Metavariables>::system;
      evolution::dg::Actions::detail::
          unit_normal_vector_and_covector_and_magnitude<system>(
              make_not_null(&normal_covectors_and_magnitudes),
              make_not_null(&fields_on_face), direction,
              unnormalized_normal_covectors, grid_to_inertial_map);
    }
    auto& outward_normal_on_sliced_mesh =
        get<evolution::dg::Tags::NormalCovector<Dim>>(
            normal_covectors_and_magnitudes[direction].value());

    Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>>
        buffer{};
    buffer.initialize(num_sliced_pts);

    // Reverse the normal component of velocity.
    auto& normal_dot_velocity = get<::Tags::TempScalar<0>>(buffer);
    auto& sliced_interior_velocity = get<Velocity>(sliced_prim_vars);
    dot_product(make_not_null(&normal_dot_velocity),
                outward_normal_on_sliced_mesh, sliced_interior_velocity);
    for (size_t i = 0; i < Dim; i++) {
      sliced_interior_velocity.get(i) -=
          2.0 * get(normal_dot_velocity) * outward_normal_on_sliced_mesh.get(i);
    }

    if (volume_mesh_velocity.has_value()) {
      auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
      dot_product(make_not_null(&normal_dot_mesh_velocity),
                  outward_normal_on_sliced_mesh, volume_mesh_velocity.value());
      for (size_t i = 0; i < Dim; i++) {
        // See the class documentation above for plus(+) sign here
        sliced_interior_velocity.get(i) += 2.0 * get(normal_dot_mesh_velocity) *
                                           outward_normal_on_sliced_mesh.get(i);
      }
    }

    // Now using OrientationMap, we reverse the data ordering of
    // sliced_prim_vars, which completes the 'mirroring' operation to get the
    // desired FD ghost data.
    std::array<Direction<Dim>, Dim> default_orientation;
    std::array<Direction<Dim>, Dim> reflected_orientation;

    if constexpr (Dim == 1) {
      default_orientation = {{Direction<1>::upper_xi()}};
      reflected_orientation = {{Direction<1>::lower_xi()}};
    } else if constexpr (Dim == 2) {
      default_orientation = {
          {Direction<2>::upper_xi(), Direction<2>::upper_eta()}};
      reflected_orientation = default_orientation;

      const size_t dim_to_reflect = direction.dimension();
      reflected_orientation[dim_to_reflect] =
          default_orientation[dim_to_reflect].opposite();
    } else if constexpr (Dim == 3) {
      default_orientation = {{Direction<3>::upper_xi(),
                              Direction<3>::upper_eta(),
                              Direction<3>::upper_zeta()}};
      reflected_orientation = default_orientation;

      const size_t dim_to_reflect = direction.dimension();
      reflected_orientation[dim_to_reflect] =
          default_orientation[dim_to_reflect].opposite();
    }

    OrientationMap<Dim> orientation_map(default_orientation,
                                        reflected_orientation);
    auto reflected_vars =
        orient_variables(sliced_prim_vars, sliced_extents, orientation_map);

    *mass_density = get<MassDensity>(reflected_vars);
    *velocity = get<Velocity>(reflected_vars);
    *pressure = get<Pressure>(reflected_vars);
  }
};

}  // namespace NewtonianEuler::BoundaryConditions
