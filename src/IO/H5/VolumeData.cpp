// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/VolumeData.hpp"

#include <algorithm>
#include <array>
#include <boost/algorithm/string.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <hdf5.h>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/ExtendConnectivityHelpers.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/SpectralIo.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/Type.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/ExpectsAndEnsures.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdHelpers.hpp"

namespace h5 {
namespace {
// Append the element extents and connectivity to the total extents and
// connectivity
constexpr const char* global_functions_of_time_observation_value_attr =
    "global_functions_of_time_observation_value";

size_t append_element_extents_and_connectivity(
    const gsl::not_null<std::vector<size_t>*> total_extents,
    const gsl::not_null<std::vector<int>*> total_connectivity,
    const gsl::not_null<int*> total_points_so_far, const size_t dim,
    const ElementVolumeData& element) {
  size_t cell_count = 0;
  // Process the element extents
  // `dim` is the dimension of the computation except when we are doing Cartoon
  // method with a 2D computational domain. In such a case, the true dimension
  // is 3 but we are writing as 2D for ParaView visualization
  const auto& extents = [&element, &dim]() {
    if (dim == 2 and element.extents.size() == 3) {
      ASSERT(gsl::at(element.basis, 2) == Spectral::Basis::Cartoon and
                 gsl::at(element.basis, 1) != Spectral::Basis::Cartoon,
             "Trying to write data with mismatched dimensions (dim = 2, "
                 << "extents = [" << element.extents[0] << ", "
                 << element.extents[1] << ", " << element.extents[2]
                 << "]) without a valid Cartoon basis (the computational "
                    "dimension must be 2).");
      return std::vector<size_t>(element.extents.begin(),
                                 element.extents.end() - 1);
    } else {
      return element.extents;
    }
  }();
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < extents.size(); ++i) {
    // The Cartoon check is for the 1D Cartoon case
    ASSERT(extents[i] != 1 or element.basis[i] == Spectral::Basis::Cartoon,
           "Cannot generate connectivity for any single grid point elements "
           "that don't use a Cartoon basis.");
  }
#endif  // SPECTRE_DEBUG
  if (extents.size() != dim) {
    ERROR("Trying to write data of dimensionality "
          << extents.size() << " but the VolumeData file has dimensionality "
          << dim << ".");
  }
  total_extents->insert(total_extents->end(), extents.begin(), extents.end());
  // Find the number of points in the local connectivity
  const int element_num_points =
      alg::accumulate(extents, 1, std::multiplies<>{});
  // Generate the connectivity data for the element
  // Possible optimization: local_connectivity.reserve(BLAH) if we can figure
  // out size without computing all the connectivities.
  const std::vector<int> connectivity = [&extents, &total_points_so_far,
                                         &cell_count]() {
    std::vector<int> local_connectivity;
    for (const auto& cell : vis::detail::compute_cells(extents)) {
      local_connectivity.emplace_back(
          vis::detail::xdmf_topology_type(cell.topology));
      ++cell_count;
      for (const auto& bounding_indices : cell.bounding_indices) {
        local_connectivity.emplace_back(*total_points_so_far +
                                        static_cast<int>(bounding_indices));
      }
    }
    return local_connectivity;
  }();
  // Capture the point offset for this element before incrementing. All
  // extra connectivity added below (for special element types) must add
  // this offset to convert local point indices to global ones.
  const int element_start = *total_points_so_far;
  *total_points_so_far += element_num_points;
  total_connectivity->insert(total_connectivity->end(), connectivity.begin(),
                             connectivity.end());

  // 2D elements may require extra connections to close periodic/angular
  // boundaries and fill any degenerate central region based on basis;
  // generically do nothing
  if (dim == 2) {
    if (element.basis[0] == Spectral::Basis::SphericalHarmonic and
        element.basis[1] == Spectral::Basis::SphericalHarmonic) {
      // Extents are (l+1, 2l+1)
      const int l = static_cast<int>(element.extents[0] - 1);

      // Connect max(phi) and min(phi) by adding more quads
      // to total_connectivity
      for (int j = 0; j < l; ++j) {
        total_connectivity->push_back(
            vis::detail::xdmf_topology_type(vis::detail::Topology::Quad));
        ++cell_count;
        total_connectivity->push_back(element_start + j);
        total_connectivity->push_back(element_start + j + 1);
        total_connectivity->push_back(element_start + 2 * l * (l + 1) + j + 1);
        total_connectivity->push_back(element_start + (2 * l) * (l + 1) + j);
      }

      // Add a new connectivity output for filling the poles
      // First, get the points at min(theta), which define the
      // boundary of the top pole to fill, and the points at
      // max(theta), which define the boundary of the bottom
      // pole to fill. Note: points are stored with theta
      // varying faster than phi.
      std::vector<int> top_pole_points{};
      std::vector<int> bottom_pole_points{};
      for (int k = 0; k < (2 * l + 1); ++k) {
        top_pole_points.push_back(element_start + k * (l + 1));
        bottom_pole_points.push_back(element_start + k * (l + 1) + l);
      }

      const size_t number_of_pole_points = top_pole_points.size();
      if (number_of_pole_points < 3) {
        ERROR_NO_TRACE(
            "Cannot write a 2D surface to file with l=0. Must have at least "
            "l=1.");
      }

      // Fill poles with triangles in a fan pattern. Choose the root point that
      // is common with all triangles to be the first point for each pole
      const int top_root_point = top_pole_points[0];
      const int bottom_root_point = bottom_pole_points[0];

      // We end such that the last triangle we make has indices (0, N-2, N-1)
      // given number_of_pole_points = N
      for (size_t i = 1; i <= number_of_pole_points - 2; i++) {
        const int top_second_point = gsl::at(top_pole_points, i);
        const int top_third_point = gsl::at(top_pole_points, i + 1);
        const int bottom_second_point = gsl::at(bottom_pole_points, i);
        const int bottom_third_point = gsl::at(bottom_pole_points, i + 1);

        total_connectivity->push_back(
            vis::detail::xdmf_topology_type(vis::detail::Topology::Triangle));
        ++cell_count;
        total_connectivity->push_back(top_root_point);
        total_connectivity->push_back(top_second_point);
        total_connectivity->push_back(top_third_point);
        total_connectivity->push_back(
            vis::detail::xdmf_topology_type(vis::detail::Topology::Triangle));
        ++cell_count;
        total_connectivity->push_back(bottom_root_point);
        total_connectivity->push_back(bottom_second_point);
        total_connectivity->push_back(bottom_third_point);
      }
    } else if ((element.basis[0] == Spectral::Basis::ZernikeB2 and
                element.basis[1] == Spectral::Basis::ZernikeB2) or
               element.basis[1] == Spectral::Basis::Fourier) {
      ASSERT(element.basis[0] == Spectral::Basis::ZernikeB2 or
                 (element.basis[0] == Spectral::Basis::Legendre or
                  element.basis[0] == Spectral::Basis::Chebyshev),
             "Adding connectivity for Fourier in the second dimension requires "
             "the first dimension to be Legendre or Chebychev, got "
                 << element.basis[0]);
      const auto n_r = static_cast<int>(extents[0]);
      const auto n_phi = static_cast<int>(extents[1]);

      // Connect max(phi) and min(phi) by adding more quads
      // to total_connectivity
      for (int j = 0; j < n_r - 1; ++j) {
        total_connectivity->push_back(
            vis::detail::xdmf_topology_type(vis::detail::Topology::Quad));
        ++cell_count;
        total_connectivity->push_back(element_start + j);
        total_connectivity->push_back(element_start + j + 1);
        total_connectivity->push_back(element_start + (n_phi - 1) * n_r + j +
                                      1);
        total_connectivity->push_back(element_start + (n_phi - 1) * n_r + j);
      }

      // For a filled disk (ZernikeB2), also fill the central hole with
      // triangles using a recursive fan pattern over the minimum r ring points.
      if (element.basis[0] == Spectral::Basis::ZernikeB2) {
        std::vector<int> inner_ring_points{};
        inner_ring_points.reserve(static_cast<size_t>(n_phi));
        for (int k = 0; k < n_phi; ++k) {
          // minimum r for each phi slice
          inner_ring_points.push_back(element_start + k * n_r);
        }

        std::vector<int> new_points;
        while (inner_ring_points.size() >= 3) {
          new_points.clear();
          new_points.push_back(inner_ring_points[0]);
          for (size_t i = 0; i < inner_ring_points.size() - 2; i += 2) {
            total_connectivity->push_back(vis::detail::xdmf_topology_type(
                vis::detail::Topology::Triangle));
            ++cell_count;
            total_connectivity->push_back(inner_ring_points[i]);
            total_connectivity->push_back(inner_ring_points[i + 1]);
            total_connectivity->push_back(inner_ring_points[i + 2]);
            new_points.push_back(inner_ring_points[i + 2]);
          }
          if (inner_ring_points.size() % 2 == 0) {
            // Add triangle closing the ring: connects last two points back to
            // first
            total_connectivity->push_back(vis::detail::xdmf_topology_type(
                vis::detail::Topology::Triangle));
            ++cell_count;
            total_connectivity->push_back(
                inner_ring_points[inner_ring_points.size() - 2]);
            total_connectivity->push_back(
                inner_ring_points[inner_ring_points.size() - 1]);
            total_connectivity->push_back(inner_ring_points[0]);
          }
          inner_ring_points = std::move(new_points);
        }
      }
    }
    // generically do nothing if not a sphere, disk, or annulus
  } else if (dim == 3) {
    if ((element.basis[0] == Spectral::Basis::ZernikeB2 and
         element.basis[1] == Spectral::Basis::ZernikeB2) or
        element.basis[1] == Spectral::Basis::Fourier) {
      ASSERT(element.basis[0] == Spectral::Basis::ZernikeB2 or
                 ((element.basis[0] == Spectral::Basis::Legendre or
                   element.basis[0] == Spectral::Basis::Chebyshev) and
                  (element.basis[2] == Spectral::Basis::Legendre or
                   element.basis[2] == Spectral::Basis::Chebyshev)),
             "Adding connectivity for Fourier in the second dimension requires "
             "the first and third dimensions to be Legendre or Chebyshev, got "
                 << element.basis[0] << ", " << element.basis[2]);
      const auto n_r = static_cast<int>(extents[0]);
      const auto n_ph = static_cast<int>(extents[1]);
      const auto n_z = static_cast<int>(extents[2]);

      // Helper: global point index for (index_r, index_phi, index_z)
      const auto global_index = [&](const int index_radius, const int index_phi,
                                    const int index_z) -> int {
        return element_start + index_radius + n_r * index_phi +
               n_r * n_ph * index_z;
      };

      // Close the phi seam: connect last phi strip (index_ph = n_ph-1) back to
      // first (index_ph = 0) with hexahedra.
      for (int j_r = 0; j_r < n_r - 1; ++j_r) {
        for (int j_z = 0; j_z < n_z - 1; ++j_z) {
          total_connectivity->push_back(vis::detail::xdmf_topology_type(
              vis::detail::Topology::Hexahedron));
          ++cell_count;
          total_connectivity->push_back(global_index(j_r, n_ph - 1, j_z));
          total_connectivity->push_back(global_index(j_r + 1, n_ph - 1, j_z));
          total_connectivity->push_back(global_index(j_r + 1, 0, j_z));
          total_connectivity->push_back(global_index(j_r, 0, j_z));
          total_connectivity->push_back(global_index(j_r, n_ph - 1, j_z + 1));
          total_connectivity->push_back(
              global_index(j_r + 1, n_ph - 1, j_z + 1));
          total_connectivity->push_back(global_index(j_r + 1, 0, j_z + 1));
          total_connectivity->push_back(global_index(j_r, 0, j_z + 1));
        }
      }

      // For a filled cylinder (ZernikeB2), fill the central hole with
      // wedges between consecutive z layers.
      if (element.basis[0] == Spectral::Basis::ZernikeB2) {
        std::vector<int> ring_lo{};
        std::vector<int> ring_hi{};
        ring_lo.reserve(static_cast<size_t>(n_ph));
        ring_hi.reserve(static_cast<size_t>(n_ph));
        std::vector<int> new_lo{};
        std::vector<int> new_hi{};
        for (int j_z = 0; j_z < n_z - 1; ++j_z) {
          ring_lo.clear();
          ring_hi.clear();
          // Collect the minimum r ring points in this z layer and the next
          for (int k = 0; k < n_ph; ++k) {
            ring_lo.push_back(global_index(0, k, j_z));
            ring_hi.push_back(global_index(0, k, j_z + 1));
          }

          // Build wedge prisms using the same recursive fan as the 2D case,
          // extruded between ring_lo and ring_hi.
          while (ring_lo.size() >= 3) {
            new_lo.clear();
            new_hi.clear();
            new_lo.push_back(ring_lo[0]);
            new_hi.push_back(ring_hi[0]);
            for (size_t i = 0; i < ring_lo.size() - 2; i += 2) {
              // Wedge (triangular prism): bottom triangle (ring_lo) + top
              // triangle (ring_hi), each 3 vertices = 6 total.
              total_connectivity->push_back(vis::detail::xdmf_topology_type(
                  vis::detail::Topology::Wedge));
              ++cell_count;
              total_connectivity->push_back(ring_lo[i]);
              total_connectivity->push_back(ring_lo[i + 1]);
              total_connectivity->push_back(ring_lo[i + 2]);
              total_connectivity->push_back(ring_hi[i]);
              total_connectivity->push_back(ring_hi[i + 1]);
              total_connectivity->push_back(ring_hi[i + 2]);
              new_lo.push_back(ring_lo[i + 2]);
              new_hi.push_back(ring_hi[i + 2]);
            }
            if (ring_lo.size() % 2 == 0) {
              // Closing wedge connecting last two points back to first
              const size_t size = ring_lo.size();
              total_connectivity->push_back(vis::detail::xdmf_topology_type(
                  vis::detail::Topology::Wedge));
              ++cell_count;
              total_connectivity->push_back(ring_lo[size - 2]);
              total_connectivity->push_back(ring_lo[size - 1]);
              total_connectivity->push_back(ring_lo[0]);
              total_connectivity->push_back(ring_hi[size - 2]);
              total_connectivity->push_back(ring_hi[size - 1]);
              total_connectivity->push_back(ring_hi[0]);
            }
            ring_lo = std::move(new_lo);
            ring_hi = std::move(new_hi);
          }
        }
      }
    }
    // If element is a 3D spherical shell (SphericalHarmonic in theta and phi
    // directions), add phi-wrapping hexahedra and pole-cap wedges.
    if (element.basis[1] == Spectral::Basis::SphericalHarmonic and
        element.basis[2] == Spectral::Basis::SphericalHarmonic) {
      const auto n_r = static_cast<int>(element.extents[0]);
      const auto n_theta = static_cast<int>(element.extents[1]);
      const auto n_phi = static_cast<int>(element.extents[2]);
      // Global point index: local index (r fastest, then theta, then phi)
      // plus the offset for this element's first point.
      const auto idx = [&n_r, &n_theta, &element_start](
                           const int ir, const int it, const int ip) -> int {
        return element_start + ir + n_r * it + n_r * n_theta * ip;
      };

      // Step 1: Add phi-wrapping hexahedra to close the phi=2*pi boundary
      for (int it = 0; it < n_theta - 1; ++it) {
        for (int ir = 0; ir < n_r - 1; ++ir) {
          total_connectivity->push_back(vis::detail::xdmf_topology_type(
              vis::detail::Topology::Hexahedron));
          ++cell_count;
          total_connectivity->push_back(idx(ir, it, n_phi - 1));
          total_connectivity->push_back(idx(ir + 1, it, n_phi - 1));
          total_connectivity->push_back(idx(ir + 1, it + 1, n_phi - 1));
          total_connectivity->push_back(idx(ir, it + 1, n_phi - 1));
          total_connectivity->push_back(idx(ir, it, 0));
          total_connectivity->push_back(idx(ir + 1, it, 0));
          total_connectivity->push_back(idx(ir + 1, it + 1, 0));
          total_connectivity->push_back(idx(ir, it + 1, 0));
        }
      }

      // Step 2: Add pole-cap wedges using recursive halving of the pole ring.
      // For each pole (theta=0 and theta=n_theta-1) and each radial layer, the
      // ring of n_phi points at fixed (ir, it_pole) is recursively halved to
      // produce Wedge cells (bottom tri at ir, top tri at ir+1).
      //
      // Winding order: phi increases CCW when viewed from the north (+z). For
      // the top pole (it_pole=0) the forward ring order gives an
      // outward-pointing bottom-triangle normal (positive signed volume). For
      // the bottom pole (it_pole=n_theta-1) the same phi order produces an
      // inward-pointing normal, so we reverse the ring to keep consistent
      // winding and positive signed volume for all wedges.
      for (const int it_pole : {0, n_theta - 1}) {
        const bool reverse_ring = (it_pole == n_theta - 1);
        for (int ir = 0; ir < n_r - 1; ++ir) {
          std::vector<int> bottom_ring;
          std::vector<int> top_ring;
          bottom_ring.reserve(static_cast<size_t>(n_phi));
          top_ring.reserve(static_cast<size_t>(n_phi));
          for (int ip = 0; ip < n_phi; ++ip) {
            bottom_ring.push_back(idx(ir, it_pole, ip));
            top_ring.push_back(idx(ir + 1, it_pole, ip));
          }
          if (reverse_ring) {
            std::reverse(bottom_ring.begin(), bottom_ring.end());
            std::reverse(top_ring.begin(), top_ring.end());
          }
          std::vector<int> new_bottom;
          std::vector<int> new_top;
          while (bottom_ring.size() >= 3) {
            new_bottom.clear();
            new_top.clear();
            new_bottom.push_back(bottom_ring[0]);
            new_top.push_back(top_ring[0]);
            for (size_t i = 0; i < bottom_ring.size() - 2; i += 2) {
              total_connectivity->push_back(vis::detail::xdmf_topology_type(
                  vis::detail::Topology::Wedge));
              ++cell_count;
              total_connectivity->push_back(bottom_ring[i]);
              total_connectivity->push_back(bottom_ring[i + 1]);
              total_connectivity->push_back(bottom_ring[i + 2]);
              total_connectivity->push_back(top_ring[i]);
              total_connectivity->push_back(top_ring[i + 1]);
              total_connectivity->push_back(top_ring[i + 2]);
              new_bottom.push_back(bottom_ring[i + 2]);
              new_top.push_back(top_ring[i + 2]);
            }
            if (bottom_ring.size() % 2 == 0) {
              const size_t sz = bottom_ring.size();
              total_connectivity->push_back(vis::detail::xdmf_topology_type(
                  vis::detail::Topology::Wedge));
              ++cell_count;
              total_connectivity->push_back(bottom_ring[sz - 2]);
              total_connectivity->push_back(bottom_ring[sz - 1]);
              total_connectivity->push_back(bottom_ring[0]);
              total_connectivity->push_back(top_ring[sz - 2]);
              total_connectivity->push_back(top_ring[sz - 1]);
              total_connectivity->push_back(top_ring[0]);
            }
            bottom_ring = std::move(new_bottom);
            top_ring = std::move(new_top);
          }
        }
      }
    }
    // generically do nothing if not a cylinder or spherical shell
  }
  return cell_count;
}
}  // namespace

VolumeData::VolumeData(const bool subfile_exists, detail::OpenGroup&& group,
                       const hid_t /*location*/, const std::string& name,
                       const uint32_t version)
    : group_(std::move(group)),
      name_(name.size() > extension().size()
                ? (extension() == name.substr(name.size() - extension().size())
                       ? name
                       : name + extension())
                : name + extension()),
      path_(group_.group_path_with_trailing_slash() + name),
      version_(version),
      volume_data_group_(group_.id(), name_, h5::AccessType::ReadWrite) {
  if (subfile_exists) {
    // We treat this as an internal version for now. We'll need to deal with
    // proper versioning later.
    const Version open_version(true, detail::OpenGroup{},
                               volume_data_group_.id(), "version");
    version_ = open_version.get_version();
    const Header header(true, detail::OpenGroup{}, volume_data_group_.id(),
                        "header");
    header_ = header.get_header();
  } else {  // file does not exist
    // Subfiles are closed as they go out of scope, so we have the extra
    // braces here to add the necessary scope
    {
      Version open_version(false, detail::OpenGroup{}, volume_data_group_.id(),
                           "version", version_);
    }
    {
      Header header(false, detail::OpenGroup{}, volume_data_group_.id(),
                    "header");
      header_ = header.get_header();
    }
  }
}

// Write Volume Data stored in a vector of `ElementVolumeData` to
// an `observation_group` in a `VolumeData` file.
void VolumeData::write_volume_data(
    const size_t observation_id, const double observation_value,
    const std::vector<ElementVolumeData>& elements,
    const std::optional<std::vector<char>>& serialized_domain,
    const std::optional<std::vector<char>>&
        serialized_observation_functions_of_time,
    const std::optional<std::vector<char>>&
        serialized_global_functions_of_time) {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  if (contains_attribute(observation_group.id(), "", "observation_value")) {
    ERROR_NO_TRACE("Trying to write ObservationId "
                   << std::to_string(observation_id)
                   << " with observation_value " << observation_group.id()
                   << " which already exists in file at " << path
                   << ". Did you forget to clean up after an earlier run?");
  }
  h5::write_to_attribute(observation_group.id(), "observation_value",
                         observation_value);
  // Get first element to extract the component names and dimension
  const auto get_component_name = [](const auto& component) {
    ASSERT(component.name.find_last_of('/') == std::string::npos,
           "The expected format of the tensor component names is "
           "'COMPONENT_NAME' but found a '/' in '"
               << component.name << "'.");
    return component.name;
  };
  const std::vector<std::string> component_names(
      boost::make_transform_iterator(elements.front().tensor_components.begin(),
                                     get_component_name),
      boost::make_transform_iterator(elements.front().tensor_components.end(),
                                     get_component_name));
  // The dimension of the grid is the number of extents per element. I.e., if
  // the extents are [8,5,7] for any element, the dimension of the grid is 3.
  // Only written once per VolumeData file (All volume data in a single file
  // should have the same dimensionality)
  if (not contains_attribute(volume_data_group_.id(), "", "dimension")) {
    h5::write_to_attribute(
        volume_data_group_.id(), "dimension",
        // Need to manually reduce dimensions with a Cartoon evolution
        // using a 2d computational domain
        [&elements]() -> size_t {
          const auto& basis = elements.front().basis;
          if (basis.size() == 3 and
              (gsl::at(basis, 2) == Spectral::Basis::Cartoon and
               gsl::at(basis, 1) != Spectral::Basis::Cartoon)) {
            return 2;
          }
          return elements.front().extents.size();
        }());
  }
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  // Extract Tensor Data one component at a time
  std::vector<size_t> total_extents;
  std::string grid_names;
  std::vector<int> total_connectivity;
  std::vector<int> quadratures;
  std::vector<int> bases;
  std::vector<uint64_t> element_ids;
  std::vector<uint64_t> block_ids;
  // Keep a running count of the number of points so far to use as a global
  // index for the connectivity
  int total_points_so_far = 0;
  // Loop over tensor components
  for (size_t i = 0; i < component_names.size(); i++) {
    std::string component_name = component_names[i];
    // Write the data for the tensor component
    if (h5::contains_dataset_or_group(observation_group.id(), "",
                                      component_name)) {
      ERROR("Trying to write tensor component '"
            << component_name
            << "' which already exists in HDF5 file in group '" << name_ << '/'
            << "ObservationId" << std::to_string(observation_id) << "'");
    }

    const auto fill_and_write_contiguous_tensor_data =
        [&bases, &block_ids, &component_name, &dim, &element_ids, &elements,
         &grid_names, i, &observation_group, &quadratures, &total_connectivity,
         &total_extents,
         &total_points_so_far](const auto contiguous_tensor_data_ptr) {
          for (const auto& element : elements) {
            if (UNLIKELY(i == 0)) {
              // True if first tensor component being accessed
              grid_names += element.element_name + h5::VolumeData::separator();
              // append element basis
              alg::transform(
                  // Need to ensure size == dim (Cartoon method with 2d
                  // computational domain needs to drop last dimension)
                  std::vector<Spectral::Basis>(
                      element.basis.begin(),
                      element.basis.end() -
                          static_cast<int>(element.basis.size() - dim)),
                  std::back_inserter(bases), [](const Spectral::Basis t) {
                    // Shift the basis to keep compatibility with old file
                    // formats.
                    return static_cast<int>(static_cast<uint8_t>(t) >>
                                            Spectral::basis_shift);
                  });
              // append element quadrature
              alg::transform(
                  // Need to ensure size == dim (Cartoon method with 2d
                  // computational domain needs to drop last dimension)
                  std::vector<Spectral::Quadrature>(
                      element.quadrature.begin(),
                      element.quadrature.end() -
                          static_cast<int>(element.basis.size() - dim)),
                  std::back_inserter(quadratures),
                  [](const Spectral::Quadrature t) {
                    return static_cast<int>(t);
                  });

              const size_t number_of_cells =
                  append_element_extents_and_connectivity(
                      &total_extents, &total_connectivity,
                      &total_points_so_far, dim, element);

              // Element ID: hash of the element name string
              if (element.element_name.size() >= 3 and
                  element.element_name[0] == '[' and
                  element.element_name[1] == 'B') {
                if (dim == 1) {
                  const ElementId<1> element_id{element.element_name};
                  element_ids.insert(element_ids.end(), number_of_cells,
                                     element_id.to_short_id());
                  block_ids.insert(block_ids.end(), number_of_cells,
                                   element_id.block_id());
                } else if (dim == 2) {
                  const ElementId<2> element_id{element.element_name};
                  element_ids.insert(element_ids.end(), number_of_cells,
                                     element_id.to_short_id());
                  block_ids.insert(block_ids.end(), number_of_cells,
                                   element_id.block_id());
                } else if (dim == 3) {
                  const ElementId<3> element_id{element.element_name};
                  element_ids.insert(element_ids.end(), number_of_cells,
                                     element_id.to_short_id());
                  block_ids.insert(block_ids.end(), number_of_cells,
                                   element_id.block_id());
                } else {
                  ERROR("Can only encode ElementID when dim is 1, 2, or 3, got "
                        << dim);
                }
              } else {
                element_ids.insert(
                    element_ids.end(), number_of_cells,
                    std::hash<std::string>{}(element.element_name));
                block_ids.insert(block_ids.end(), number_of_cells, 0);
              }
            }
            using type_from_variant = tmpl::conditional_t<
                std::is_same_v<
                    std::decay_t<decltype(*contiguous_tensor_data_ptr)>,
                    std::vector<double>>,
                DataVector, std::vector<float>>;
            const auto& tensor_component = element.tensor_components[i];
            ASSERT(tensor_component.name == component_name,
                   "Tensor components must be in the same order for all "
                   "elements. Expected '"
                       << component_name << "' but found '"
                       << tensor_component.name << "' at index " << i << ".");
            contiguous_tensor_data_ptr->insert(
                contiguous_tensor_data_ptr->end(),
                std::get<type_from_variant>(tensor_component.data).begin(),
                std::get<type_from_variant>(tensor_component.data).end());
          }  // for each element
          h5::write_data(observation_group.id(), *contiguous_tensor_data_ptr,
                         {contiguous_tensor_data_ptr->size()}, component_name);
        };

    if (elements[0].tensor_components[i].data.index() == 0) {
      std::vector<double> contiguous_tensor_data{};
      fill_and_write_contiguous_tensor_data(
          make_not_null(&contiguous_tensor_data));
    } else if (elements[0].tensor_components[i].data.index() == 1) {
      std::vector<float> contiguous_tensor_data{};
      fill_and_write_contiguous_tensor_data(
          make_not_null(&contiguous_tensor_data));
    } else {
      ERROR("Unknown index value ("
            << elements[0].tensor_components[i].data.index()
            << ") in std::variant of tensor component.");
    }
  }  // for each component
  grid_names.pop_back();

  // Write the grid extents contiguously, the first `dim` belong to the
  // First grid, the second `dim` belong to the second grid, and so on,
  // Ordering is `x, y, z, ... `
  h5::write_data(observation_group.id(), total_extents, {total_extents.size()},
                 "total_extents");
  // Write the names of the grids as vector of chars with individual names
  // separated by `separator()`
  std::vector<char> grid_names_as_chars(grid_names.begin(), grid_names.end());
  h5::write_data(observation_group.id(), grid_names_as_chars,
                 {grid_names_as_chars.size()}, "grid_names");
  // Write the coded quadrature, along with the dictionary
  const auto io_quadratures = Spectral::all_quadratures();
  std::vector<std::string> quadrature_dict(io_quadratures.size());
  alg::transform(io_quadratures, quadrature_dict.begin(),
                 get_output<Spectral::Quadrature>);
  h5_detail::write_dictionary("Quadrature dictionary", quadrature_dict,
                              observation_group);
  h5::write_data(observation_group.id(), quadratures, {quadratures.size()},
                 "quadratures");
  // Write the coded basis, along with the dictionary
  const auto io_bases = Spectral::all_bases();
  std::vector<std::string> basis_dict(io_bases.size());
  alg::transform(io_bases, basis_dict.begin(), get_output<Spectral::Basis>);
  h5_detail::write_dictionary("Basis dictionary", basis_dict,
                              observation_group);
  h5::write_data(observation_group.id(), bases, {bases.size()}, "bases");
  // Write the Connectivity, which will only be empty when a CartoonSphere
  // domain is used, which has 1 computational dimension and should not be
  // visualized with ParaView
  if (not total_connectivity.empty()) {
    h5::write_data(observation_group.id(), total_connectivity,
                   {total_connectivity.size()}, "connectivity");
  }
  // Write cell-centered element_id and block_id datasets for the mixed
  // topology format. element_id is the hash of the element name; block_id is
  // parsed from the "[B<N>,..." element name pattern.
  if (not element_ids.empty()) {
    h5::write_data(observation_group.id(), element_ids, {element_ids.size()},
                   "ElementId");
  }
  if (not block_ids.empty()) {
    h5::write_data(observation_group.id(), block_ids, {block_ids.size()},
                   "BlockId");
  }
  // Store the serialized domain and functions of time at the subfile level
  if (serialized_domain.has_value() and
      not contains_dataset_or_group(volume_data_group_.id(), "", "domain")) {
    h5::write_data(volume_data_group_.id(), *serialized_domain,
                   {serialized_domain->size()}, "domain");
  }
  if (serialized_observation_functions_of_time.has_value()) {
    h5::write_data(observation_group.id(),
                   *serialized_observation_functions_of_time,
                   {serialized_observation_functions_of_time->size()},
                   "functions_of_time");
  }
  if (serialized_global_functions_of_time.has_value()) {
    bool should_write_global_functions_of_time = true;
    if (h5::contains_attribute(
            volume_data_group_.id(), "",
            global_functions_of_time_observation_value_attr)) {
      const auto stored_observation_value = h5::read_value_attribute<double>(
          volume_data_group_.id(),
          global_functions_of_time_observation_value_attr);
      should_write_global_functions_of_time =
          observation_value > stored_observation_value;
    }
    if (should_write_global_functions_of_time) {
      h5::write_data(volume_data_group_.id(),
                     *serialized_global_functions_of_time,
                     {serialized_global_functions_of_time->size()},
                     "global_functions_of_time", true);
      if (h5::contains_attribute(
              volume_data_group_.id(), "",
              global_functions_of_time_observation_value_attr)) {
        CHECK_H5(H5Adelete(volume_data_group_.id(),
                           global_functions_of_time_observation_value_attr),
                 "Failed to delete existing attribute '"
                     << global_functions_of_time_observation_value_attr << "'");
      }
      h5::write_to_attribute(volume_data_group_.id(),
                             global_functions_of_time_observation_value_attr,
                             observation_value);
    }
  }
}

// Write new connectivity connections given a std::vector of observation ids
template <size_t SpatialDim>
void VolumeData::extend_connectivity_data(
    const std::vector<size_t>& observation_ids) {
  for (const size_t& obs_id : observation_ids) {
    auto grid_names = get_grid_names(obs_id);
    auto extents = get_extents(obs_id);
    auto bases = get_bases(obs_id);
    auto quadratures = get_quadratures(obs_id);

    const std::vector<int>& new_connectivity =
        h5::detail::extend_connectivity<SpatialDim>(grid_names, bases,
                                                    quadratures, extents);

    // Deletes the existing connectivity and replaces it with the new one
    const std::string path = "ObservationId" + std::to_string(obs_id);
    detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                        AccessType::ReadWrite);
    const hid_t group_id = observation_group.id();
    delete_connectivity(group_id);
    write_connectivity(group_id, new_connectivity);
  }
}

void VolumeData::write_tensor_component(
    const size_t observation_id, const std::string& component_name,
    const DataVector& contiguous_tensor_data, const bool overwrite_existing) {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  h5::write_data(observation_group.id(), contiguous_tensor_data, component_name,
                 overwrite_existing);
}

void VolumeData::write_tensor_component(
    const size_t observation_id, const std::string& component_name,
    const std::vector<float>& contiguous_tensor_data,
    const bool overwrite_existing) {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadWrite);
  h5::write_data(observation_group.id(), contiguous_tensor_data,
                 {contiguous_tensor_data.size()}, component_name,
                 overwrite_existing);
}

bool VolumeData::has_domain() const {
  return contains_dataset_or_group(volume_data_group_.id(), "", "domain");
}

bool VolumeData::has_global_functions_of_time() const {
  return contains_dataset_or_group(volume_data_group_.id(), "",
                                   "global_functions_of_time");
}

std::vector<size_t> VolumeData::list_observation_ids() const {
  const auto names = get_group_names(volume_data_group_.id(), "");
  std::vector<size_t> obs_ids{};
  obs_ids.reserve(names.size());
  constexpr std::string_view observation_prefix{"ObservationId"};
  for (const auto& name : names) {
    if (name.size() <= observation_prefix.size() or
        name.compare(0, observation_prefix.size(), observation_prefix) != 0) {
      continue;
    }
    obs_ids.push_back(std::stoul(name.substr(observation_prefix.size())));
  }
  // pre-compute the observation values as they are expensive to evaluate
  std::unordered_map<size_t, double> obs_values{obs_ids.size()};
  for (const auto& id : obs_ids) {
    obs_values[id] = this->get_observation_value(id);
  }
  alg::sort(obs_ids, [&obs_values](const size_t lhs, const size_t rhs) {
    return obs_values[lhs] < obs_values[rhs];
  });
  return obs_ids;
}

double VolumeData::get_observation_value(const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  return h5::read_value_attribute<double>(observation_group.id(),
                                          "observation_value");
}

size_t VolumeData::find_observation_id(
    const double observation_value,
    const std::optional<double>& observation_value_epsilon) const {
  std::optional<size_t> result_observation_id{};
  for (const size_t observation_id : list_observation_ids()) {
    const double file_observation_value = get_observation_value(observation_id);
    // If we are given an epsilon, use that to compare within roundoff using the
    // requested value as a scale (unless it's zero, then do 1.0). If we aren't
    // given an epsilon, compare exactly.
    if ((observation_value_epsilon.has_value()
             ? equal_within_roundoff(
                   observation_value, file_observation_value,
                   observation_value_epsilon.value(),
                   (observation_value == 0.0 ? 1.0 : observation_value))
             : observation_value == file_observation_value)) {
      if (result_observation_id.has_value()) {
        ERROR_NO_TRACE("There are multiple observations with the same value "
                       << observation_value << " within an epsilon of "
                       << observation_value_epsilon.value_or(
                              std::numeric_limits<double>::epsilon())
                       << " in the volume file " << name_);
      }
      result_observation_id = observation_id;
    }
  }
  if (not result_observation_id.has_value()) {
    ERROR_NO_TRACE("No observation with value " << observation_value
                                                << " found in volume file.");
  }

  return result_observation_id.value();
}

std::vector<std::string> VolumeData::list_tensor_components(
    const size_t observation_id) const {
  auto tensor_components =
      get_group_names(volume_data_group_.id(),
                      "ObservationId" + std::to_string(observation_id));
  // Remove names that are not tensor components
  const std::unordered_set<std::string> non_tensor_components{
      "connectivity",
      "pole_connectivity",
      "tetrahedral_connectivity",
      "total_extents",
      "grid_names",
      "quadratures",
      "bases",
      "domain",
      "functions_of_time",
      "ElementId",
      "BlockId"};
  tensor_components.erase(
      alg::remove_if(tensor_components,
                     [&non_tensor_components](const std::string& name) {
                       return non_tensor_components.find(name) !=
                              non_tensor_components.end();
                     }),
      tensor_components.end());
  return tensor_components;
}

std::vector<std::string> VolumeData::get_grid_names(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const std::vector<char> names =
      h5::read_data<1, std::vector<char>>(observation_group.id(), "grid_names");
  const std::string all_names(names.begin(), names.end());
  std::vector<std::string> grid_names{};
  boost::split(grid_names, all_names,
               [](const char c) { return c == h5::VolumeData::separator(); });
  return grid_names;
}

TensorComponent VolumeData::get_tensor_component(
    const size_t observation_id, const std::string& tensor_component) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);

  const hid_t dataset_id =
      h5::open_dataset(observation_group.id(), tensor_component);
  const hid_t dataspace_id = h5::open_dataspace(dataset_id);
  const auto rank =
      static_cast<size_t>(H5Sget_simple_extent_ndims(dataspace_id));
  h5::close_dataspace(dataspace_id);
  const auto h5_data_type = H5Dget_type(dataset_id);
  h5::close_dataset(dataset_id);

  const auto get_data = [&observation_group, &rank,
                         &tensor_component](auto type_to_get_v) {
    using type_to_get = tmpl::type_from<decltype(type_to_get_v)>;
    switch (rank) {
      case 1:
        return h5::read_data<1, type_to_get>(observation_group.id(),
                                             tensor_component);
      case 2:
        return h5::read_data<2, type_to_get>(observation_group.id(),
                                             tensor_component);
      case 3:
        return h5::read_data<3, type_to_get>(observation_group.id(),
                                             tensor_component);
      default:
        ERROR("Rank must be 1, 2, or 3. Received data with Rank = " << rank);
    }
  };

  if (h5::types_equal(h5_data_type, h5::h5_type<float>())) {
    return {tensor_component, get_data(tmpl::type_<std::vector<float>>{})};
  } else if (h5::types_equal(h5_data_type, h5::h5_type<double>())) {
    return {tensor_component, get_data(tmpl::type_<DataVector>{})};
  } else if (h5::types_equal(h5_data_type, h5::h5_type<int>())) {
    const std::vector<int> stored = get_data(tmpl::type_<std::vector<int>>{});
    DataVector result{stored.size()};
    std::ranges::copy(stored.begin(), stored.end(), result.begin());
    return {tensor_component, result};
  } else if (h5::types_equal(h5_data_type, h5::h5_type<unsigned int>())) {
    const std::vector<unsigned int> stored =
        get_data(tmpl::type_<std::vector<unsigned int>>{});
    DataVector result{stored.size()};
    std::ranges::copy(stored.begin(), stored.end(), result.begin());
    return {tensor_component, result};
  } else if (h5::types_equal(h5_data_type, h5::h5_type<unsigned long>())) {
    const std::vector<unsigned long> stored =
        get_data(tmpl::type_<std::vector<unsigned long>>{});
    DataVector result{stored.size()};
    std::ranges::copy(stored.begin(), stored.end(), result.begin());
    return {tensor_component, result};
  } else {
    ERROR("Unknown H5 type " << h5_data_type);
  }
}

std::vector<std::vector<size_t>> VolumeData::get_extents(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto extents_per_element = static_cast<long>(dim);
  const auto total_extents = h5::read_data<1, std::vector<size_t>>(
      observation_group.id(), "total_extents");
  std::vector<std::vector<size_t>> individual_extents;
  individual_extents.reserve(total_extents.size() / dim);
  for (auto iter = total_extents.begin(); iter != total_extents.end();
       iter += extents_per_element) {
    individual_extents.emplace_back(iter, iter + extents_per_element);
  }
  return individual_extents;
}

std::pair<size_t, size_t> offset_and_length_for_grid(
    const std::string& grid_name,
    const std::vector<std::string>& all_grid_names,
    const std::vector<std::vector<size_t>>& all_extents) {
  auto found_grid_name = alg::find(all_grid_names, grid_name);
  if (found_grid_name == all_grid_names.end()) {
    ERROR("Found no grid named '" + grid_name + "'.");
  } else {
    const auto element_index =
        std::distance(all_grid_names.begin(), found_grid_name);
    const size_t element_data_offset = std::accumulate(
        all_extents.begin(), all_extents.begin() + element_index, 0_st,
        [](const size_t offset, const std::vector<size_t>& extents) {
          return offset + alg::accumulate(extents, 1_st, std::multiplies<>{});
        });
    const size_t element_data_length = alg::accumulate(
        gsl::at(all_extents, element_index), 1_st, std::multiplies<>{});
    return {element_data_offset, element_data_length};
  }
}

auto VolumeData::get_data_by_element(
    const std::optional<double> start_observation_value,
    const std::optional<double> end_observation_value,
    const std::optional<std::vector<std::string>>& components_to_retrieve) const
    -> std::vector<std::tuple<size_t, double, std::vector<ElementVolumeData>>> {
  // First get list of all observations we need to retrieve
  const std::vector<size_t> obs_ids = list_observation_ids();
  std::vector<std::tuple<size_t, double, std::vector<ElementVolumeData>>>
      result{};
  result.reserve(obs_ids.size());
  // Sort observation IDs and observation values into the result. This only
  // copies observed times in
  // [`start_observation_value`, `end_observation_value`]
  for (const auto& observation_id : obs_ids) {
    const double observation_value = get_observation_value(observation_id);
    if (start_observation_value.value_or(
            std::numeric_limits<double>::lowest()) <= observation_value and
        observation_value <= end_observation_value.value_or(
                                 std::numeric_limits<double>::max())) {
      result.emplace_back(observation_id, observation_value,
                          std::vector<ElementVolumeData>{});
    }
  }
  result.shrink_to_fit();
  // Sort by observation_value
  alg::sort(result, [](const auto& lhs, const auto& rhs) {
    return std::get<1>(lhs) < std::get<1>(rhs);
  });

  // Retrieve element data and insert into result
  for (auto& single_time_data : result) {
    const auto known_components =
        list_tensor_components(std::get<0>(single_time_data));

    std::vector<ElementVolumeData> element_volume_data{};
    const auto grid_names = get_grid_names(std::get<0>(single_time_data));
    const auto extents = get_extents(std::get<0>(single_time_data));
    const auto bases = get_bases(std::get<0>(single_time_data));
    const auto quadratures = get_quadratures(std::get<0>(single_time_data));
    element_volume_data.reserve(grid_names.size());

    const auto& component_names =
        components_to_retrieve.value_or(known_components);
    std::vector<TensorComponent> tensors{};
    tensors.reserve(grid_names.size());
    for (const std::string& component : component_names) {
      if (not alg::found(known_components, component)) {
        using ::operator<<;  // STL streams
        ERROR("Could not find tensor component '"
              << component
              << "' in file. Known components are: " << known_components);
      }
      tensors.emplace_back(
          get_tensor_component(std::get<0>(single_time_data), component));
    }
    // Now split the data by element
    for (size_t grid_index = 0, offset = 0; grid_index < grid_names.size();
         ++grid_index) {
      const size_t mesh_size =
          alg::accumulate(extents[grid_index], 1_st, std::multiplies<>{});
      std::vector<TensorComponent> tensor_components{tensors.size()};
      for (size_t component_index = 0; component_index < tensors.size();
           ++component_index) {
        std::visit(
            [component_index, &component_names, mesh_size, offset,
             &tensor_components](const auto& tensor_component_data) {
              std::decay_t<decltype(tensor_component_data)> component(
                  mesh_size);
              std::copy(
                  std::next(tensor_component_data.begin(),
                            static_cast<std::ptrdiff_t>(offset)),
                  std::next(tensor_component_data.begin(),
                            static_cast<std::ptrdiff_t>(offset + mesh_size)),
                  component.begin());
              tensor_components[component_index] = TensorComponent{
                  component_names[component_index], std::move(component)};
            },
            tensors[component_index].data);
      }

      // Sort the tensor components by name so that they are in the same order
      // in all elements.
      alg::sort(tensor_components, [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
      });

      element_volume_data.emplace_back(
          grid_names[grid_index], std::move(tensor_components),
          extents[grid_index], bases[grid_index], quadratures[grid_index]);
      offset += mesh_size;
    }  // for grid_index

    // Sort the elements so they are in the same order at all time steps
    alg::sort(element_volume_data,
              [](const ElementVolumeData& lhs, const ElementVolumeData& rhs) {
                return lhs.element_name < rhs.element_name;
              });
    std::get<2>(single_time_data) = std::move(element_volume_data);
  }
  return result;
}

size_t VolumeData::get_dimension() const {
  return h5::read_value_attribute<double>(volume_data_group_.id(), "dimension");
}

std::vector<std::vector<Spectral::Basis>> VolumeData::get_bases(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto bases_per_element = static_cast<long>(dim);

  const std::vector<int> bases_coded =
      h5::read_data<1, std::vector<int>>(observation_group.id(), "bases");
  const auto all_bases = h5_detail::decode_with_dictionary_name(
      "Basis dictionary", bases_coded, observation_group);

  std::vector<std::vector<Spectral::Basis>> element_bases;
  for (auto iter = all_bases.begin(); iter != all_bases.end();
       std::advance(iter, bases_per_element)) {
    element_bases.emplace_back(
        boost::make_transform_iterator(iter, Spectral::to_basis),
        boost::make_transform_iterator(std::next(iter, bases_per_element),
                                       Spectral::to_basis));
  }
  return element_bases;
}
std::vector<std::vector<Spectral::Quadrature>> VolumeData::get_quadratures(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  const auto dim =
      h5::read_value_attribute<size_t>(volume_data_group_.id(), "dimension");
  const auto quadratures_per_element = static_cast<long>(dim);
  const std::vector<int> quadratures_coded =
      h5::read_data<1, std::vector<int>>(observation_group.id(), "quadratures");
  const auto all_quadratures = h5_detail::decode_with_dictionary_name(
      "Quadrature dictionary", quadratures_coded, observation_group);
  std::vector<std::vector<Spectral::Quadrature>> element_quadratures;
  for (auto iter = all_quadratures.begin(); iter != all_quadratures.end();
       std::advance(iter, quadratures_per_element)) {
    element_quadratures.emplace_back(
        boost::make_transform_iterator(iter, Spectral::to_quadrature),
        boost::make_transform_iterator(std::next(iter, quadratures_per_element),
                                       Spectral::to_quadrature));
  }
  return element_quadratures;
}

std::optional<std::vector<char>> VolumeData::get_domain() const {
  // we write the domain independently of the observation_id since a refactor
  if (contains_dataset_or_group(volume_data_group_.id(), "", "domain")) {
    return h5::read_data<1, std::vector<char>>(volume_data_group_.id(),
                                               "domain");
  }
  // default to old location
  const auto& observation_ids = list_observation_ids();
  if (observation_ids.empty()) {
    return std::nullopt;
  }
  const size_t observation_id = observation_ids.back();
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  if (contains_dataset_or_group(observation_group.id(), "", "domain")) {
    return h5::read_data<1, std::vector<char>>(observation_group.id(),
                                               "domain");
  }
  return std::nullopt;
}

std::optional<std::vector<char>> VolumeData::get_functions_of_time(
    const size_t observation_id) const {
  const std::string path = "ObservationId" + std::to_string(observation_id);
  detail::OpenGroup observation_group(volume_data_group_.id(), path,
                                      AccessType::ReadOnly);
  if (not contains_dataset_or_group(observation_group.id(), "",
                                    "functions_of_time")) {
    return std::nullopt;
  }
  return h5::read_data<1, std::vector<char>>(observation_group.id(),
                                             "functions_of_time");
}

std::optional<std::vector<char>> VolumeData::get_global_functions_of_time()
    const {
  if (contains_dataset_or_group(volume_data_group_.id(), "",
                                "global_functions_of_time")) {
    return h5::read_data<1, std::vector<char>>(volume_data_group_.id(),
                                               "global_functions_of_time");
  }
  // fall back to old location if global not present
  const auto& observation_ids = list_observation_ids();
  if (observation_ids.empty()) {
    return std::nullopt;
  }
  return get_functions_of_time(observation_ids.back());
}

template <size_t Dim>
Mesh<Dim> mesh_for_grid(
    const std::string& grid_name,
    const std::vector<std::string>& all_grid_names,
    const std::vector<std::vector<size_t>>& all_extents,
    const std::vector<std::vector<Spectral::Basis>>& all_bases,
    const std::vector<std::vector<Spectral::Quadrature>>& all_quadratures) {
  const auto found_grid_name = alg::find(all_grid_names, grid_name);
  if (found_grid_name == all_grid_names.end()) {
    ERROR("Found no grid named '" + grid_name + "'.");
  } else {
    const auto element_index =
        std::distance(all_grid_names.begin(), found_grid_name);
    const auto& extents = gsl::at(all_extents, element_index);
    const auto& bases = gsl::at(all_bases, element_index);
    const auto& quadratures = gsl::at(all_quadratures, element_index);
    ASSERT(extents.size() == Dim, "Extents in " << Dim << "D should have size "
                                                << Dim << ", but found size "
                                                << extents.size() << ".");
    ASSERT(bases.size() == Dim, "Bases in " << Dim << "D should have size "
                                            << Dim << ", but found size "
                                            << bases.size() << ".");
    ASSERT(quadratures.size() == Dim, "Quadratures in "
                                          << Dim << "D should have size " << Dim
                                          << ", but found size "
                                          << quadratures.size() << ".");
    return Mesh<Dim>{make_array<size_t, Dim>(extents),
                     make_array<Spectral::Basis, Dim>(bases),
                     make_array<Spectral::Quadrature, Dim>(quadratures)};
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                         \
  template void h5::VolumeData::extend_connectivity_data<DIM(data)>( \
      const std::vector<size_t>& observation_ids);                   \
  template Mesh<DIM(data)> mesh_for_grid(                            \
      const std::string& grid_name,                                  \
      const std::vector<std::string>& all_grid_names,                \
      const std::vector<std::vector<size_t>>& all_extents,           \
      const std::vector<std::vector<Spectral::Basis>>& all_bases,    \
      const std::vector<std::vector<Spectral::Quadrature>>& all_quadratures);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace h5
