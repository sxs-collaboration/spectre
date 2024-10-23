// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/NumericInitialData.hpp"
#include "IO/External/InterpolateFromCocal.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

enum class CocalIdType { Co, Ir, Sp };

/*!
 * \brief Hydro initial data generated by COCAL.
 *
 * This class loads numerical data written out by the COCAL initial data solver.
 *
 * We choose a constant electron fraction and zero temperature for now because
 * COCAL doesn't export these quantities. We'll have to improve this later, e.g.
 * by constructing an EOS consistent with the COCAL data.
 */
class CocalInitialData : public evolution::initial_data::InitialData,
                         public evolution::NumericInitialData,
                         public AnalyticDataBase {
 public:
  struct DataDirectory {
    using type = std::string;
    static constexpr Options::String help = {
        "Path to the directory of data produced by COCAL."};
  };

  struct ElectronFraction {
    using type = double;
    static constexpr Options::String help = {"Constant electron fraction"};
  };

  struct IdType {
    using type = CocalIdType;
    static constexpr Options::String help = {
        "The ID type of COCAL data (Co, Ir, Sp)"};
  };
  using options = tmpl::list<DataDirectory, ElectronFraction, IdType>;

  static constexpr Options::String help = {"Initial data generated by COCAL"};

  CocalInitialData() = default;
  CocalInitialData(const CocalInitialData& rhs);
  CocalInitialData& operator=(const CocalInitialData& rhs);
  CocalInitialData(CocalInitialData&& rhs);
  CocalInitialData& operator=(CocalInitialData&& rhs);
  ~CocalInitialData() override = default;

  CocalInitialData(std::string data_directory, double electron_fraction,
                   CocalIdType id_type);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit CocalInitialData(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CocalInitialData);
  /// \endcond

  template <typename DataType>
  using tags = tmpl::append<
      tmpl::list<gr::Tags::SpatialMetric<DataType, 3>,
                 gr::Tags::ExtrinsicCurvature<DataType, 3>,
                 gr::Tags::Lapse<DataType>, gr::Tags::Shift<DataType, 3>>,
      hydro::grmhd_tags<DataType>>;

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    // Pass data_directory_ and id_type_ to the interpolation function
    auto interpolated_vars = io::interpolate_from_cocal(
        make_not_null(&cocal_lock_), static_cast<io::CocalIdType>(id_type_),
        data_directory_, x);
    tuples::TaggedTuple<RequestedTags...> result{};
    // Compute derived quantities from interpolated data
    const size_t num_points = x.begin()->size();

    // Move interpolated data into result buffer
    tmpl::for_each<
        tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                   gr::Tags::SpatialMetric<DataVector, 3>,
                   gr::Tags::ExtrinsicCurvature<DataVector, 3>,
                   hydro::Tags::RestMassDensity<DataVector>,
                   hydro::Tags::SpecificInternalEnergy<DataVector>,
                   hydro::Tags::Pressure<DataVector>,
                   hydro::Tags::LorentzFactor<DataVector>,
                   hydro::Tags::SpatialVelocity<DataVector, 3>>>(
        [&result, &interpolated_vars](const auto tag_v) {
          using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
          get<tag>(result) = std::move(get<tag>(interpolated_vars));
        });
    // Compute derived quantities from interpolated data
    // const size_t num_points = x.begin()->size();
    // const auto& rest_mass_density =
    //     get<hydro::Tags::RestMassDensity<DataVector>>(result);
    const auto& specific_internal_energy =
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(result);
    const auto& spatial_velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, 3>>(result);
    const auto& pressure = get<hydro::Tags::Pressure<DataVector>>(result);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, 3>>(result);
    // Compute enthalpy from internal energy and pressure
    // auto& specific_enthalpy =
    //     get<hydro::Tags::SpecificEnthalpy<DataVector>>(result);
    // get(specific_enthalpy) = DataVector(num_points);
    // for (size_t i = 0; i < num_points; ++i) {
    //   const double local_rest_mass_density = get(rest_mass_density)[i];
    //   if (equal_within_roundoff(local_rest_mass_density, 0.)) {
    //     get(specific_enthalpy)[i] = 1.;
    //   } else {
    //     get(specific_enthalpy)[i] =
    //     get(hydro::relativistic_specific_enthalpy(
    //         Scalar<double>(local_rest_mass_density),
    //         Scalar<double>(get(specific_internal_energy)[i]),
    //         Scalar<double>(get(pressure)[i])));

    // Constant electron fraction specified by input file
    auto& electron_fraction =
        get<hydro::Tags::ElectronFraction<DataVector>>(result);
    get(electron_fraction) = DataVector(num_points, electron_fraction_);
    // Zero magnetic field and divergence cleaning field
    auto& magnetic_field =
        get<hydro::Tags::MagneticField<DataVector, 3>>(result);
    get<0>(magnetic_field) = DataVector(num_points, 0.);
    get<1>(magnetic_field) = DataVector(num_points, 0.);
    get<2>(magnetic_field) = DataVector(num_points, 0.);
    auto& div_cleaning_field =
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(result);
    get(div_cleaning_field) = DataVector(num_points, 0.);
    // Compute Lorentz factor from spatial velocity
    auto& lorentz_factor = get<hydro::Tags::LorentzFactor<DataVector>>(result);

    get(lorentz_factor) =
        1. / sqrt(1. - get(dot_product(spatial_velocity, spatial_velocity,
                                       spatial_metric)));


    // Set temperature to zero for now
    auto& temperature = get<hydro::Tags::Temperature<DataVector>>(result);
    get(temperature) = DataVector(num_points, 0.);
    return result;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  std::string data_directory_{};
  CocalIdType id_type_;
  double electron_fraction_ = std::numeric_limits<double>::signaling_NaN();

  // This lock is used to ensure that only one thread at a time is calling the
  // COCAL interpolation routines. We make some assumptions here to guarantee
  // thread-safety:
  // - This analytic data class exists only once per node (in the global cache).
  //   This means we don't have to copy or PUP the lock or pass it around
  //   instances.
  // - This also allows the lock to be mutable, which is necessary for the
  //   const-ness of the `variables` function.
  mutable std::mutex cocal_lock_{};  // NOLINT(spectre-mutable)
};

}  // namespace grmhd::AnalyticData

// Specialization for YAML parsing
namespace Options {
template <>
struct create_from_yaml<grmhd::AnalyticData::CocalIdType> {
  template <typename Metavariables>
  static grmhd::AnalyticData::CocalIdType create(
      const Options::Option& options) {
    const std::string& id_type_str = options.parse_as<std::string>();
    if (id_type_str == "Co") {
      return grmhd::AnalyticData::CocalIdType::Co;
    } else if (id_type_str == "Ir") {
      return grmhd::AnalyticData::CocalIdType::Ir;
    } else if (id_type_str == "Sp") {
      return grmhd::AnalyticData::CocalIdType::Sp;
    } else {
      PARSE_ERROR(options.context(),
                  "Failed to convert \""
                      << id_type_str
                      << "\" to CocalIdType. Expected one of: Co, Ir, Sp.");
    }
  }
};
}  // namespace Options
