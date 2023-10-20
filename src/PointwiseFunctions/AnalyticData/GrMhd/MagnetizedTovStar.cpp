// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::AnalyticData {
MagnetizedTovStar::MagnetizedTovStar() = default;
MagnetizedTovStar::MagnetizedTovStar(MagnetizedTovStar&& /*rhs*/) = default;
MagnetizedTovStar& MagnetizedTovStar::operator=(MagnetizedTovStar&& /*rhs*/) =
    default;
MagnetizedTovStar::~MagnetizedTovStar() = default;

MagnetizedTovStar::MagnetizedTovStar(
    const double central_rest_mass_density,
    std::unique_ptr<MagnetizedTovStar::equation_of_state_type>
        equation_of_state,
    const RelativisticEuler::Solutions::TovCoordinates coordinate_system,
    std::vector<std::unique_ptr<
        grmhd::AnalyticData::InitialMagneticFields::InitialMagneticField>>
        magnetic_fields)
    : tov_star(central_rest_mass_density, std::move(equation_of_state),
               coordinate_system),
      magnetic_fields_(std::move(magnetic_fields)) {}

MagnetizedTovStar::MagnetizedTovStar(const MagnetizedTovStar& rhs)
    : evolution::initial_data::InitialData{rhs},
      RelativisticEuler::Solutions::TovStar(
          static_cast<const RelativisticEuler::Solutions::TovStar&>(rhs)),
      magnetic_fields_(clone_unique_ptrs(rhs.magnetic_fields_)) {}

MagnetizedTovStar& MagnetizedTovStar::operator=(const MagnetizedTovStar& rhs) {
  if (this == &rhs) {
    return *this;
  }
  static_cast<RelativisticEuler::Solutions::TovStar&>(*this) =
      static_cast<const RelativisticEuler::Solutions::TovStar&>(rhs);
  magnetic_fields_ = clone_unique_ptrs(rhs.magnetic_fields_);
  return *this;
}

std::unique_ptr<evolution::initial_data::InitialData>
MagnetizedTovStar::get_clone() const {
  return std::make_unique<MagnetizedTovStar>(*this);
}

MagnetizedTovStar::MagnetizedTovStar(CkMigrateMessage* msg) : tov_star(msg) {}

void MagnetizedTovStar::pup(PUP::er& p) {
  tov_star::pup(p);
  p | magnetic_fields_;
}

namespace magnetized_tov_detail {
template <typename DataType, StarRegion Region>
void MagnetizedTovVariables<DataType, Region>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
    const gsl::not_null<Cache*> cache,
    hydro::Tags::MagneticField<DataType, 3> /*meta*/) const {
  const Scalar<DataType>& sqrt_det_spatial_metric =
      cache->get_var(*this, gr::Tags::SqrtDetSpatialMetric<DataType>{});
  const Scalar<DataType>& pressure =
      cache->get_var(*this, hydro::Tags::Pressure<DataType>{});
  const auto& deriv_pressure =
      cache->get_var(*this, ::Tags::deriv<hydro::Tags::Pressure<DataType>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  set_number_of_grid_points(magnetic_field, get_size(get<0>(coords)));
  for (size_t i = 0; i < 3; ++i) {
    magnetic_field->get(i) = 0.0;
  }

  for (const auto& magnetic_field_compute : magnetic_fields) {
    magnetic_field_compute->variables(magnetic_field, coords, pressure,
                                      sqrt_det_spatial_metric, deriv_pressure);
  }
}
}  // namespace magnetized_tov_detail

PUP::able::PUP_ID MagnetizedTovStar::my_PUP_ID = 0;

bool operator==(const MagnetizedTovStar& lhs, const MagnetizedTovStar& rhs) {
  bool equal =
      static_cast<const typename MagnetizedTovStar::tov_star&>(lhs) ==
          static_cast<const typename MagnetizedTovStar::tov_star&>(rhs) and
      lhs.magnetic_fields_.size() == rhs.magnetic_fields_.size();
  if (not equal) {
    return false;
  }
  for (size_t i = 0; i < lhs.magnetic_fields_.size(); ++i) {
    if (not lhs.magnetic_fields_[i]->is_equal(*rhs.magnetic_fields_[i])) {
      return false;
    }
  }
  return true;
}

bool operator!=(const MagnetizedTovStar& lhs, const MagnetizedTovStar& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define REGION(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template class magnetized_tov_detail::MagnetizedTovVariables<DTYPE(data), \
                                                               REGION(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (magnetized_tov_detail::StarRegion::Center,
                         magnetized_tov_detail::StarRegion::Interior,
                         magnetized_tov_detail::StarRegion::Exterior))

#undef INSTANTIATE
#undef DTYPE
#undef REGION
}  // namespace grmhd::AnalyticData
