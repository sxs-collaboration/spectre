// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <iterator>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/NeighborData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::fd {
template <typename TagsList, size_t ThermodynamicDim>
void compute_conservatives_for_reconstruction(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        eos) {
  // Computes:
  // 1. W v^i
  // 2. Lorentz factor as sqrt(1 + Wv^i Wv^j\gamma_{ij})
  // 3. v^i = Wv^i / W
  // 4. specific internal energy
  // 5. specific enthalpy
  // 6. conserved variables
  // - note: spatial metric, inv spatial metric, lapse, and shift are
  //         all already in vars_on_face
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(*vars_on_face);
  const auto& lorentz_factor_times_spatial_velocity =
      get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
          *vars_on_face);
  auto& lorentz_factor =
      get<hydro::Tags::LorentzFactor<DataVector>>(*vars_on_face);
  get(lorentz_factor) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    get(lorentz_factor) += spatial_metric.get(i, i) *
                           square(lorentz_factor_times_spatial_velocity.get(i));
    for (size_t j = i + 1; j < 3; ++j) {
      get(lorentz_factor) += 2.0 * spatial_metric.get(i, j) *
                             lorentz_factor_times_spatial_velocity.get(i) *
                             lorentz_factor_times_spatial_velocity.get(j);
    }
  }
  get(lorentz_factor) = sqrt(1.0 + get(lorentz_factor));
  auto& spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(*vars_on_face) =
          lorentz_factor_times_spatial_velocity;
  for (size_t i = 0; i < 3; ++i) {
    spatial_velocity.get(i) /= get(lorentz_factor);
  }
  const auto& rest_mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(*vars_on_face);
  const auto& pressure = get<hydro::Tags::Pressure<DataVector>>(*vars_on_face);
  auto& specific_internal_energy =
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(*vars_on_face);
  if constexpr (ThermodynamicDim == 2) {
    specific_internal_energy =
        eos.specific_internal_energy_from_density_and_pressure(
            rest_mass_density, pressure);
  } else {
    specific_internal_energy =
        eos.specific_internal_energy_from_density(rest_mass_density);
  }
  auto& specific_enthalpy =
      get<hydro::Tags::SpecificEnthalpy<DataVector>>(*vars_on_face);
  hydro::relativistic_specific_enthalpy(make_not_null(&specific_enthalpy),
                                        rest_mass_density,
                                        specific_internal_energy, pressure);
  ConservativeFromPrimitive::apply(
      make_not_null(&get<ValenciaDivClean::Tags::TildeD>(*vars_on_face)),
      make_not_null(&get<ValenciaDivClean::Tags::TildeTau>(*vars_on_face)),
      make_not_null(
          &get<ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(*vars_on_face)),
      make_not_null(
          &get<ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(*vars_on_face)),
      make_not_null(&get<ValenciaDivClean::Tags::TildePhi>(*vars_on_face)),
      rest_mass_density, specific_internal_energy, specific_enthalpy, pressure,
      spatial_velocity, lorentz_factor,
      get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
          *vars_on_face),
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(*vars_on_face),
      get<gr::Tags::SpatialMetric<3>>(*vars_on_face),
      get<hydro::Tags::DivergenceCleaningField<DataVector>>(*vars_on_face));
}

template <typename PrimsTags, typename TagsList, size_t ThermodynamicDim,
          typename F>
void reconstruct_prims_work(
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    const gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const F& reconstruct, const Variables<PrimsTags>& volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<maximum_number_of_neighbors(3) + 1,
                       std::pair<Direction<3>, ElementId<3>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<3>, ElementId<3>>>>
        neighbor_data,
    const Mesh<3>& subcell_mesh, size_t ghost_zone_size) {
  using prim_tags_for_reconstruction =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  ASSERT(Mesh<3>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                 subcell_mesh.quadrature(0)) == subcell_mesh,
         "The subcell mesh should be isotropic but got " << subcell_mesh);
  const size_t volume_num_pts = subcell_mesh.number_of_grid_points();
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  const size_t neighbor_num_pts =
      ghost_zone_size * subcell_mesh.extents().slice_away(0).product();
  size_t vars_in_neighbor_count = 0;
  tmpl::for_each<prim_tags_for_reconstruction>([&element, &neighbor_data,
                                                neighbor_num_pts, &reconstruct,
                                                reconstructed_num_pts,
                                                volume_num_pts, &volume_prims,
                                                &vars_in_neighbor_count,
                                                &vars_on_lower_face,
                                                &vars_on_upper_face,
                                                &subcell_mesh](
                                                   auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const typename tag::type* volume_tensor_ptr = nullptr;
    Variables<tmpl::list<
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>>
        lorentz_factor_times_v_I{};
    if constexpr (std::is_same_v<tag,
                                 hydro::Tags::LorentzFactorTimesSpatialVelocity<
                                     DataVector, 3>>) {
      // we need to handle the Wv^i reconstruction separately since we need to
      // first compute Wv^i in the volume (it's not one of our primitives from
      // the recovery). The components need to be stored contiguously, which is
      // why we have the Variables `lorentz_factor_times_v_I`
      const auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(volume_prims);
      const auto& lorentz_factor =
          get<hydro::Tags::LorentzFactor<DataVector>>(volume_prims);
      lorentz_factor_times_v_I.initialize(get(lorentz_factor).size());
      auto& volume_tensor =
          get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
              lorentz_factor_times_v_I) = spatial_velocity;
      for (size_t i = 0; i < 3; ++i) {
        volume_tensor.get(i) *= get(lorentz_factor);
      }
      volume_tensor_ptr = &volume_tensor;
    } else {
      volume_tensor_ptr = &get<tag>(volume_prims);
    }

    const size_t number_of_variables = volume_tensor_ptr->size();
    const gsl::span<const double> volume_vars = gsl::make_span(
        (*volume_tensor_ptr)[0].data(), number_of_variables * volume_num_pts);
    std::array<gsl::span<double>, 3> upper_face_vars{};
    std::array<gsl::span<double>, 3> lower_face_vars{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(upper_face_vars, i) =
          gsl::make_span(get<tag>(gsl::at(*vars_on_upper_face, i))[0].data(),
                         number_of_variables * reconstructed_num_pts);
      gsl::at(lower_face_vars, i) =
          gsl::make_span(get<tag>(gsl::at(*vars_on_lower_face, i))[0].data(),
                         number_of_variables * reconstructed_num_pts);
    }

    DirectionMap<3, gsl::span<const double>> ghost_cell_vars{};
    for (const auto& direction : Direction<3>::all_directions()) {
      const auto& neighbors_in_direction = element.neighbors().at(direction);
      ASSERT(neighbors_in_direction.size() == 1,
             "Currently only support one neighbor in each direction, but "
             "got "
                 << neighbors_in_direction.size() << " in direction "
                 << direction);
      ghost_cell_vars[direction] = gsl::make_span(
          &neighbor_data
               .at(std::pair{direction, *neighbors_in_direction.begin()})
               .data_for_reconstruction[vars_in_neighbor_count *
                                        neighbor_num_pts],
          number_of_variables * neighbor_num_pts);
    }

    reconstruct(make_not_null(&upper_face_vars),
                make_not_null(&lower_face_vars), volume_vars, ghost_cell_vars,
                subcell_mesh.extents(), number_of_variables);

    vars_in_neighbor_count += number_of_variables;
  });

  for (size_t i = 0; i < 3; ++i) {
    compute_conservatives_for_reconstruction(
        make_not_null(&gsl::at(*vars_on_lower_face, i)), eos);
    compute_conservatives_for_reconstruction(
        make_not_null(&gsl::at(*vars_on_upper_face, i)), eos);
  }
}

template <typename TagsList, typename PrimsTags, size_t ThermodynamicDim,
          typename F0, typename F1>
void reconstruct_fd_neighbor_work(
    const gsl::not_null<Variables<TagsList>*> vars_on_face,
    const F0& reconstruct_lower_neighbor, const F1& reconstruct_upper_neighbor,
    const Variables<PrimsTags>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<maximum_number_of_neighbors(3) + 1,
                       std::pair<Direction<3>, ElementId<3>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<3>, ElementId<3>>>>
        neighbor_data,
    const Mesh<3>& subcell_mesh, const Direction<3>& direction_to_reconstruct,
    const size_t ghost_zone_size) {
  using prim_tags_for_reconstruction =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  const std::pair mortar_id{
      direction_to_reconstruct,
      *element.neighbors().at(direction_to_reconstruct).begin()};
  Index<3> ghost_data_extents = subcell_mesh.extents();
  ghost_data_extents[direction_to_reconstruct.dimension()] = ghost_zone_size;
  Variables<prim_tags_for_reconstruction> neighbor_prims{
      ghost_data_extents.product()};
  {
    ASSERT(neighbor_data.contains(mortar_id),
           "The neighbor data does not contain the mortar: ("
               << mortar_id.first << ',' << mortar_id.second << ")");
    const auto& neighbor_data_on_mortar = neighbor_data.at(mortar_id);
    std::copy(neighbor_data_on_mortar.data_for_reconstruction.begin(),
              std::next(neighbor_data_on_mortar.data_for_reconstruction.begin(),
                        static_cast<std::ptrdiff_t>(
                            neighbor_prims.number_of_independent_components *
                            ghost_data_extents.product())),
              neighbor_prims.data());
  }

  tmpl::for_each<prim_tags_for_reconstruction>(
      [&direction_to_reconstruct, &ghost_data_extents, &neighbor_prims,
       &reconstruct_lower_neighbor, &reconstruct_upper_neighbor, &subcell_mesh,
       &subcell_volume_prims, &vars_on_face](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const typename tag::type* volume_tensor_ptr = nullptr;
        typename tag::type volume_tensor{};
        if constexpr (std::is_same_v<
                          tag, hydro::Tags::LorentzFactorTimesSpatialVelocity<
                                   DataVector, 3>>) {
          // we need to handle the Wv^i reconstruction separately since we need
          // to first compute Wv^i in the volume (it's not one of our primitives
          // from the recovery). The components need to be stored contiguously,
          // which is why we have the Variables `lorentz_factor_times_v_I`
          const auto& spatial_velocity =
              get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
                  subcell_volume_prims);
          const auto& lorentz_factor =
              get<hydro::Tags::LorentzFactor<DataVector>>(subcell_volume_prims);
          volume_tensor = spatial_velocity;
          for (size_t i = 0; i < 3; ++i) {
            volume_tensor.get(i) *= get(lorentz_factor);
          }
          volume_tensor_ptr = &volume_tensor;
        } else {
          volume_tensor_ptr = &get<tag>(subcell_volume_prims);
        }

        const auto& tensor_neighbor = get<tag>(neighbor_prims);
        auto& tensor_on_face = get<tag>(*vars_on_face);
        if (direction_to_reconstruct.side() == Side::Upper) {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_upper_neighbor(
                make_not_null(&tensor_on_face[tensor_index]),
                (*volume_tensor_ptr)[tensor_index],
                tensor_neighbor[tensor_index], subcell_mesh.extents(),
                ghost_data_extents, direction_to_reconstruct);
          }
        } else {
          for (size_t tensor_index = 0; tensor_index < tensor_on_face.size();
               ++tensor_index) {
            reconstruct_lower_neighbor(
                make_not_null(&tensor_on_face[tensor_index]),
                (*volume_tensor_ptr)[tensor_index],
                tensor_neighbor[tensor_index], subcell_mesh.extents(),
                ghost_data_extents, direction_to_reconstruct);
          }
        }
      });

  compute_conservatives_for_reconstruction(vars_on_face, eos);
}
}  // namespace grmhd::ValenciaDivClean::fd
