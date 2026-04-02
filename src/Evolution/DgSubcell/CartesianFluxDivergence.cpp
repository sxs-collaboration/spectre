// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/CartesianFluxDivergence.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/ElementMap.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
void add_cartesian_flux_divergence(const gsl::not_null<DataVector*> dt_var,
                                   const double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<1>& subcell_extents,
                                   const size_t dimension) {
  (void)dimension;
  ASSERT(dimension == 0, "dimension must be 0 but is " << dimension);
  for (size_t i = 0; i < subcell_extents[0]; ++i) {
    (*dt_var)[i] += one_over_delta * inv_jacobian[i] *
                    (boundary_correction[i + 1] - boundary_correction[i]);
  }
}

void add_cartesian_flux_divergence(const gsl::not_null<DataVector*> dt_var,
                                   const double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<2>& subcell_extents,
                                   const size_t dimension) {
  ASSERT(dimension == 0 or dimension == 1,
         "dimension must be 0 or 1 but is " << dimension);
  Index<2> subcell_face_extents = subcell_extents;
  ++subcell_face_extents[dimension];
  for (size_t j = 0; j < subcell_extents[1]; ++j) {
    for (size_t i = 0; i < subcell_extents[0]; ++i) {
      Index<2> index(i, j);
      const size_t volume_index = collapsed_index(index, subcell_extents);
      const size_t boundary_correction_lower_index =
          collapsed_index(index, subcell_face_extents);
      ++index[dimension];
      const size_t boundary_correction_upper_index =
          collapsed_index(index, subcell_face_extents);
      (*dt_var)[volume_index] +=
          one_over_delta * inv_jacobian[volume_index] *
          (boundary_correction[boundary_correction_upper_index] -
           boundary_correction[boundary_correction_lower_index]);
    }
  }
}

void add_cartesian_flux_divergence(const gsl::not_null<DataVector*> dt_var,
                                   const double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<3>& subcell_extents,
                                   const size_t dimension) {
  ASSERT(dimension == 0 or dimension == 1 or dimension == 2,
         "dimension must be 0, 1, or 2 but is " << dimension);
  Index<3> subcell_face_extents = subcell_extents;
  ++subcell_face_extents[dimension];
  for (size_t k = 0; k < subcell_extents[2]; ++k) {
    for (size_t j = 0; j < subcell_extents[1]; ++j) {
      for (size_t i = 0; i < subcell_extents[0]; ++i) {
        Index<3> index(i, j, k);
        const size_t volume_index = collapsed_index(index, subcell_extents);
        const size_t boundary_correction_lower_index =
            collapsed_index(index, subcell_face_extents);
        ++index[dimension];
        const size_t boundary_correction_upper_index =
            collapsed_index(index, subcell_face_extents);
        (*dt_var)[volume_index] +=
            one_over_delta * inv_jacobian[volume_index] *
            (boundary_correction[boundary_correction_upper_index] -
             boundary_correction[boundary_correction_lower_index]);
      }
    }
  }
}

void add_cartoon_cartesian_flux_divergence(
    const gsl::not_null<DataVector*> dt_var, const double one_over_delta,
    const DataVector& inv_jacobian, const DataVector& boundary_correction,
    const Index<3>& subcell_extents, const size_t dimension,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  // Validate that this is only used with cartoon bases
  ASSERT(subcell_extents[2] == 1,
         "Expecting extent = 1 in third dimension, but got extents = "
             << subcell_extents);
  ASSERT(subcell_extents[0] != 1,
         "Expecting extent != 1 in first dimension, but got extents = "
             << subcell_extents);

  Index<3> subcell_face_extents = subcell_extents;
  ++subcell_face_extents[dimension];
  const size_t num_volume_points = subcell_extents.product();
  ASSERT(inertial_coords.get(0).size() == num_volume_points,
         "inertial_coords size " << inertial_coords.get(0).size()
                                 << " does not match expected volume points "
                                 << num_volume_points);

  const bool spherical = subcell_extents[1] == 1;

  if (spherical) {
    // 2nd and 3rd dimension handled by cartoon
    ASSERT(dimension == 0,
           "Using cartoon derivatives with spherical symmetry, expecting "
           "dimension = 0, got dimension = "
               << dimension);
  } else {
    // 3rd dimension handled by cartoon
    ASSERT(dimension == 0 or dimension == 1,
           "Using cartoon derivatives with axial symmetry, expecting "
           "dimension = 0 or 1, got dimension = "
               << dimension);
  }

  if (UNLIKELY(
          equal_within_roundoff(0.0, get<0>(inertial_coords)[0],
                                std::numeric_limits<double>::epsilon() * 100.0,
                                max(get<0>(inertial_coords))))) {
    ERROR(
        "Element contains x=0 for subcell; this should not happen for cartoon "
        "FD grids");
  }

  Mesh<3> face_centered_mesh;
  if (spherical) {
    face_centered_mesh = Mesh<3>{
        {subcell_extents[0] + 1, subcell_extents[1], subcell_extents[2]},
        {Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::FaceCentered,
         Spectral::Quadrature::SphericalSymmetry,
         Spectral::Quadrature::SphericalSymmetry}};
  } else {
    face_centered_mesh = Mesh<3>{
        {subcell_extents[0] + (dimension == 0 ? 1 : 0),
         subcell_extents[1] + (dimension == 1 ? 1 : 0), subcell_extents[2]},
        {Spectral::Basis::FiniteDifference, Spectral::Basis::FiniteDifference,
         Spectral::Basis::Cartoon},
        {dimension == 0 ? Spectral::Quadrature::FaceCentered
                        : Spectral::Quadrature::CellCentered,
         dimension == 1 ? Spectral::Quadrature::FaceCentered
                        : Spectral::Quadrature::CellCentered,
         Spectral::Quadrature::AxialSymmetry}};
  }

  const auto face_centered_logical_coords =
      logical_coordinates(face_centered_mesh);
  const auto face_centered_inertial_coords =
      grid_to_inertial_map(logical_to_grid_map(face_centered_logical_coords),
                           time, functions_of_time);

  for (size_t k = 0; k < subcell_extents[2]; ++k) {
    for (size_t j = 0; j < subcell_extents[1]; ++j) {
      for (size_t i = 0; i < subcell_extents[0]; ++i) {
        Index<3> index(i, j, k);
        const size_t volume_index = collapsed_index(index, subcell_extents);
        const size_t boundary_correction_lower_index =
            collapsed_index(index, subcell_face_extents);
        ++index[dimension];
        const size_t boundary_correction_upper_index =
            collapsed_index(index, subcell_face_extents);

        double lower_face_weight{};
        double upper_face_weight{};
        if (spherical) {
          lower_face_weight = square(get<0>(face_centered_inertial_coords)
                                         [boundary_correction_lower_index]) /
                              square(get<0>(inertial_coords)[volume_index]);

          upper_face_weight = square(get<0>(face_centered_inertial_coords)
                                         [boundary_correction_upper_index]) /
                              square(get<0>(inertial_coords)[volume_index]);
        } else {
          lower_face_weight = get<0>(face_centered_inertial_coords)
                                  [boundary_correction_lower_index] /
                              get<0>(inertial_coords)[volume_index];
          upper_face_weight = get<0>(face_centered_inertial_coords)
                                  [boundary_correction_upper_index] /
                              get<0>(inertial_coords)[volume_index];
        }
        (*dt_var)[volume_index] +=
            one_over_delta * inv_jacobian[volume_index] *
            (upper_face_weight *
                 boundary_correction[boundary_correction_upper_index] -
             lower_face_weight *
                 boundary_correction[boundary_correction_lower_index]);
      }
    }
  }
}
}  // namespace evolution::dg::subcell
