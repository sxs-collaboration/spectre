// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/AnalyticData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::AnalyticData::ScalarField {

/*!
 * \brief $\frac{A}{r}$ profile for the scalar field.
 *
 * The expression of the profile as a function of the coordinates reads
 * \begin{equation}
 *   \Psi(x^i) = \frac{A}{|x^i - x^i_0|},
 * \end{equation}
 * where $\Psi$ is the scalar field, $x^i_0 = (0, 0, 0)$ is the coordinate
 * location of the center of the profile and $A$ is the amplitude. Note that the
 * norm for the distance is computed using the eucledian metric.
 */
template <size_t Dim>
class Inverser
    : public ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim> {
 public:
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = "Scaled Amplitude.";
  };

  using options = tmpl::list<Amplitude>;
  static constexpr Options::String help{
      "A/r profile for initial guess of the scalar field."};

  using scalar_field_tags =
      typename ScalarTensor::AnalyticData::ScalarField::AnalyticData<
          Dim>::scalar_field_tags;

  Inverser() = default;
  Inverser(const Inverser&) = default;
  Inverser& operator=(const Inverser&) = default;
  Inverser(Inverser&&) = default;
  Inverser& operator=(Inverser&&) = default;
  ~Inverser() override = default;

  explicit Inverser(double amplitude);

  /// \cond
  explicit Inverser(CkMigrateMessage* m)
      : ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Inverser);  // NOLINT
  /// \endcond

  std::unique_ptr<ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>>
  get_clone() const override;

  tuples::TaggedTuple<::CurvedScalarWave::Tags::Psi> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<::CurvedScalarWave::Tags::Psi> /*meta*/) const;

  tuples::TaggedTuple<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>
  variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>> /*meta*/)
      const;

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    static_assert(
        tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                         scalar_field_tags>>::value == 0,
        "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }

  void pup(PUP::er& p) override;

 private:
  double amplitude_{std::numeric_limits<double>::signaling_NaN()};

  template <size_t SpatialDim>
  friend bool operator==(const Inverser<SpatialDim>& lhs,
                         const Inverser<SpatialDim>& rhs);
};

template <size_t SpatialDim>
bool operator!=(const Inverser<SpatialDim>& lhs,
                const Inverser<SpatialDim>& rhs);

}  // namespace ScalarTensor::AnalyticData::ScalarField
