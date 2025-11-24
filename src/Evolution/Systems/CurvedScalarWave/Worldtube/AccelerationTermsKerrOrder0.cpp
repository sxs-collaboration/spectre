
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

void acceleration_terms_kerr_0(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, const double ft,
    const double fx, const double fy, const double fz, const double dt_ft,
    const double dt_fx, const double dt_fy, const double dt_fz,
    const double bh_mass, const std::array<double, 3>& bh_spin) {
  const size_t grid_size = get<0>(centered_coords).size();
  result->initialize(grid_size);
  const double xp = particle_position[0];
  const double yp = particle_position[1];
  const double zp = particle_position[2];
  const double xpdot = particle_velocity[0];
  const double ypdot = particle_velocity[1];
  const double zpdot = particle_velocity[2];

  const double xpddot = particle_acceleration[0];
  const double ypddot = particle_acceleration[1];
  const double zpddot = particle_acceleration[2];

  const double rp = get(magnitude(particle_position));
  const double rpdot = (xp * xpdot + yp * ypdot + zp * zpdot) / rp;

  const auto& Dx = get<0>(centered_coords);
  const auto& Dy = get<1>(centered_coords);
  const auto& Dz = get<2>(centered_coords);

  const double M = bh_mass;
  const double a = bh_spin[2];

  DynamicBuffer<DataVector> temps(56, grid_size);

  const double d_0 = a * a;
  const double d_1 = rp * rp;
  const double d_2 = d_0 + d_1;
  const double d_3 = 1.0 / d_2;
  const double d_4 = rp * yp;
  const double d_5 = a * xp - d_4;
  const double d_6 = d_3 * d_5;
  const double d_7 = -d_6 * ypdot;
  const double d_8 = a * yp;
  const double d_9 = rp * xp;
  const double d_10 = d_8 + d_9;
  const double d_11 = d_10 * d_3;
  const double d_12 = d_11 * xpdot;
  const double d_13 = 1.0 / rp;
  const double d_14 = d_13 * zp;
  const double d_15 = d_14 * zpdot;
  const double d_16 = d_15 + 1;
  const double d_17 = d_12 + d_16;
  const double d_18 = d_17 + d_7;
  const double d_19 = rp * rp * rp;
  const double d_20 = rp * rp * rp * rp;
  const double d_21 = zp * zp;
  const double d_22 = d_0 * d_21;
  const double d_23 = d_20 + d_22;
  const double d_24 = 1.0 / d_23;
  const double d_25 = M * d_24;
  const double d_26 = 2 * d_25;
  const double d_27 = d_19 * d_26;
  const double d_28 = xpdot * xpdot;
  const double d_29 = ypdot * ypdot;
  const double d_30 = zpdot * zpdot;
  const double d_31 = d_28 + d_29 + d_30 - 1;
  const double d_32 = d_27 * (d_18 * d_18) + d_31;
  const double d_33 = 1.0 / d_32;
  const double d_34 = d_11 * d_27;
  const double d_35 = d_5 * d_5;
  const double d_36 = 1.0 / (d_2 * d_2);
  const double d_37 = d_27 * d_36;
  const double d_38 = d_35 * d_37;
  const double d_39 = d_1 * d_26;
  const double d_40 = 2 * rp;
  const double d_41 = d_25 * d_40;
  const double d_42 = d_21 * d_41;
  const double d_43 = d_10 * d_10;
  const double d_44 = d_37 * d_43;
  const double d_45 = 4 * d_25;
  const double d_46 = d_19 * d_45;
  const double d_47 = d_39 * zp;
  const double d_48 = d_6 * fy;
  const double d_49 = (1.0 / 2.0) * d_33;
  const double d_50 = -d_5;
  const double d_51 = d_3 * d_50;
  const double d_52 = d_51 * ypdot;
  const double d_53 = d_16 + d_52;
  const double d_54 = d_12 + d_53;
  const double d_55 = d_54 * d_54;
  const double d_56 = d_27 * d_55 + d_31;
  const double d_57 = -d_56;
  const double d_58 = 1.0 / d_56;
  const double d_59 = rp * zpdot - rpdot * zp;
  const double d_60 = d_41 * d_59;
  const double d_61 = zp * zpdot;
  const double d_62 = 3 * d_22;
  const double d_63 = d_0 * d_40 * d_61 + d_20 * rpdot - d_62 * rpdot;
  const double d_64 = 1.0 / (d_23 * d_23);
  const double d_65 = M * d_64;
  const double d_66 = d_63 * d_65;
  const double d_67 = d_40 * d_66;
  const double d_68 = d_1 * d_63 * d_65;
  const double d_69 = 2 * d_68;
  const double d_70 = d_40 * rpdot;
  const double d_71 =
      -d_2 * (-a * xpdot + rp * ypdot + rpdot * yp) + d_50 * d_70;
  const double d_72 = d_10 * d_70 - d_2 * (a * ypdot + rp * xpdot + rpdot * xp);
  const double d_73 = -d_72;
  const double d_74 = d_59 * 1.0 / d_1;
  const double d_75 = -d_71;
  const double d_76 = sqrt(d_57);
  const double d_77 = 2 * xpdot;
  const double d_78 = 2 * ypdot;
  const double d_79 = 2 * zpdot;
  const double d_80 = d_20 - d_62;
  const double d_81 = xp * xp;
  const double d_82 = yp * yp;
  const double d_83 = d_21 + d_81 + d_82;
  const double d_84 = -d_0 + d_83;
  const double d_85 = sqrt(4 * d_22 + d_84 * d_84);
  const double d_86 = sqrt(d_84 + d_85);
  const double d_87 = M_SQRT2;
  const double d_88 = 1.0 / d_85;
  const double d_89 = d_87 * d_88;
  const double d_90 = d_86 * d_89;
  const double d_91 = d_90 * xp;
  const double d_92 = d_91 * xpdot;
  const double d_93 = d_90 * yp * ypdot;
  const double d_94 = 4 * rp;
  const double d_95 = d_89 * (d_0 + d_83 + d_85) * 1.0 / d_86;
  const double d_96 = d_14 * zpddot;
  const double d_97 = d_11 * xpddot;
  const double d_98 = d_51 * ypddot;
  const double d_99 = d_61 * d_95;
  const double d_100 = 2 * d_90;
  const double d_101 = d_10 * d_100;
  const double d_102 = 2 * a;
  const double d_103 = d_36 * xpdot;
  const double d_104 = d_36 * ypdot;
  const double d_105 = d_27 * d_54;
  const double d_106 =
      -M * d_1 * d_55 * d_64 *
          (d_61 * (d_0 * d_94 + d_80 * d_95) + d_80 * d_92 + d_80 * d_93) +
      d_105 *
          (d_103 * (d_99 * (-d_40 * d_8 + xp * (d_0 - d_1)) +
                    xpdot * (-d_101 * d_9 + d_2 * (d_40 + d_81 * d_90)) +
                    ypdot * (-d_101 * d_4 + d_2 * (d_102 + d_91 * yp))) +
           d_104 *
               (d_99 * (d_0 * yp - d_1 * yp + d_102 * d_9) +
                xpdot * (d_100 * d_5 * d_9 +
                         d_2 * (-d_102 + d_86 * d_87 * d_88 * xp * yp)) +
                ypdot * (-d_100 * d_4 * d_50 + d_2 * (d_40 + d_82 * d_90))) +
           d_13 * zpdot *
               (-d_14 * d_92 - d_14 * d_93 +
                zpdot * (-d_13 * d_21 * d_95 + 2)) +
           2 * d_96 + 2 * d_97 + 2 * d_98) +
      d_77 * xpddot + d_78 * ypddot + d_79 * zpddot;
  const double d_107 = 1.0 / d_57;
  const double d_108 = d_50 * d_50;
  const double d_109 = d_108 * d_37;
  const double d_110 = 2 * d_21 * d_66;
  const double d_111 = d_36 * d_43;
  const double d_112 = d_111 * d_69;
  const double d_113 = d_108 * d_36;
  const double d_114 = d_113 * d_69;
  const double d_115 = 1.0 / (d_2 * d_2 * d_2);
  const double d_116 = d_12 + 1;
  const double d_117 = d_42 * zpdot + zpdot;
  const double d_118 = d_19 * d_25;
  const double d_119 = d_111 * d_118 * d_77 + xpdot;
  const double d_120 = d_118 * d_78;
  const double d_121 = d_17 * d_27;
  const double d_122 = d_36 * d_75;
  const double d_123 = d_45 * zp;
  const double d_124 = d_115 * d_46;
  const double d_125 = -d_106;
  const double d_126 = -d_48;
  const double d_127 = d_14 * fz + ft;
  const double d_128 = d_16 + d_7;
  const double d_129 = d_11 * fx;
  const double d_130 = d_27 * d_6;
  const double d_131 = d_1 * d_123;
  DataVector& dv_0 = temps.at(0);
  dv_0 = Dy * zpdot;
  DataVector& dv_1 = temps.at(1);
  dv_1 = Dz * ypdot;
  DataVector& dv_2 = temps.at(2);
  dv_2 = d_14 * dv_1;
  DataVector& dv_3 = temps.at(3);
  dv_3 = d_27 * (Dy + d_14 * dv_0 + dv_2);
  DataVector& dv_4 = temps.at(4);
  dv_4 = Dx * ypdot;
  DataVector& dv_5 = temps.at(5);
  dv_5 = Dy * xpdot;
  DataVector& dv_6 = temps.at(6);
  dv_6 = dv_4 + dv_5;
  DataVector& dv_7 = temps.at(7);
  dv_7 = Dx * zpdot;
  DataVector& dv_8 = temps.at(8);
  dv_8 = Dz * xpdot;
  DataVector& dv_9 = temps.at(9);
  dv_9 = d_14 * dv_7 + d_14 * dv_8;
  DataVector& dv_10 = temps.at(10);
  dv_10 = Dy * ypdot;
  DataVector& dv_11 = temps.at(11);
  dv_11 = Dz * zp;
  DataVector& dv_12 = temps.at(12);
  dv_12 = d_39 * dv_11;
  DataVector& dv_13 = temps.at(13);
  dv_13 = Dz * zpdot;
  DataVector& dv_14 = temps.at(14);
  dv_14 = Dx * xpdot;
  DataVector& dv_15 = temps.at(15);
  dv_15 = dv_10 + dv_13 + dv_14;
  DataVector& dv_16 = temps.at(16);
  dv_16 = d_42 * dv_13 + d_44 * dv_14 + dv_12 + dv_15;
  DataVector& dv_17 = temps.at(17);
  dv_17 = d_38 * dv_10 + dv_16;
  DataVector& dv_18 = temps.at(18);
  dv_18 = d_34 * (Dx - d_6 * dv_6 + dv_9) - d_6 * dv_3 + dv_17;
  DataVector& dv_19 = temps.at(19);
  dv_19 = dv_18 * dv_18;
  DataVector& dv_20 = temps.at(20);
  dv_20 = Dz * d_14;
  DataVector& dv_21 = temps.at(21);
  dv_21 = Dy * d_6;
  DataVector& dv_22 = temps.at(22);
  dv_22 = -dv_20 + dv_21;
  DataVector& dv_23 = temps.at(23);
  dv_23 = Dx * d_11;
  DataVector& dv_24 = temps.at(24);
  dv_24 = d_46 * dv_23;
  DataVector& dv_25 = temps.at(25);
  dv_25 = Dy * Dy;
  DataVector& dv_26 = temps.at(26);
  dv_26 = d_1 * d_45 * dv_11;
  DataVector& dv_27 = temps.at(27);
  dv_27 = Dx * Dx;
  DataVector& dv_28 = temps.at(28);
  dv_28 = Dz * Dz;
  DataVector& dv_29 = temps.at(29);
  dv_29 = d_42 * dv_28 + d_44 * dv_27 + dv_25 + dv_27 + dv_28;
  DataVector& dv_30 = temps.at(30);
  dv_30 = d_38 * dv_25 - dv_21 * dv_26 + dv_29;
  DataVector& dv_31 = temps.at(31);
  dv_31 = -d_33 * dv_19 - dv_22 * dv_24 + dv_30;
  DataVector& dv_32 = temps.at(32);
  dv_32 = dv_20 - dv_21 + dv_23;
  DataVector& dv_33 = temps.at(33);
  dv_33 = d_27 * dv_32;
  DataVector& dv_34 = temps.at(34);
  dv_34 = d_18 * dv_33 + dv_15;
  DataVector& dv_35 = temps.at(35);
  dv_35 = dv_34 * dv_34;
  DataVector& dv_36 = temps.at(36);
  dv_36 = d_32 * dv_31 + dv_35;
  DataVector& dv_37 = temps.at(37);
  dv_37 = dv_32 * fz;
  DataVector& dv_38 = temps.at(38);
  dv_38 = Dx + d_11 * dv_33;
  DataVector& dv_39 = temps.at(39);
  dv_39 =
      Dy * fy + Dz * fz + d_47 * dv_37 - d_48 * dv_33 + dv_33 * ft + dv_38 * fx;
  DataVector& dv_40 = temps.at(40);
  dv_40 = dv_36 * dv_39;
  DataVector& dv_41 = temps.at(19);
  dv_41 = -d_58 * dv_19 - dv_22 * dv_24 + dv_30;
  DataVector& dv_42 = temps.at(35);
  dv_42 = d_56 * dv_41 + dv_35;
  DataVector& dv_43 = temps.at(30);
  dv_43 = d_69 * dv_32;
  DataVector& dv_44 = temps.at(41);
  dv_44 = Dx * d_36;
  DataVector& dv_45 = temps.at(42);
  dv_45 = Dz * d_74;
  DataVector& dv_46 = temps.at(43);
  dv_46 = Dy * d_36;
  DataVector& dv_47 = temps.at(44);
  dv_47 = d_75 * dv_46 + dv_45;
  DataVector& dv_48 = temps.at(45);
  dv_48 = -d_12 - d_15 - d_52 + d_73 * dv_44 + dv_47;
  DataVector& dv_49 = temps.at(46);
  dv_49 = Dy * d_51;
  DataVector& dv_50 = temps.at(20);
  dv_50 = dv_20 + dv_49;
  DataVector& dv_51 = temps.at(47);
  dv_51 = dv_23 + dv_50;
  DataVector& dv_52 = temps.at(48);
  dv_52 = d_27 * dv_51;
  DataVector& dv_53 = temps.at(49);
  dv_53 = d_69 * dv_51;
  DataVector& dv_54 = temps.at(19);
  dv_54 = sqrt(dv_41);
  DataVector& dv_55 = temps.at(50);
  dv_55 = dv_39 * dv_42;
  DataVector& dv_56 = temps.at(6);
  dv_56 = Dx + d_51 * dv_6;
  DataVector& dv_57 = temps.at(16);
  dv_57 = d_109 * dv_10 + d_34 * (dv_56 + dv_9) + d_51 * dv_3 + dv_16;
  DataVector& dv_58 = temps.at(3);
  dv_58 = dv_57 * dv_57;
  DataVector& dv_59 = temps.at(24);
  dv_59 = d_107 * dv_58 + d_109 * dv_25 + dv_24 * dv_50 + dv_26 * dv_49 + dv_29;
  DataVector& dv_60 = temps.at(26);
  dv_60 = d_107 * dv_57;
  DataVector& dv_61 = temps.at(29);
  dv_61 = d_27 * dv_44;
  DataVector& dv_62 = temps.at(9);
  dv_62 = Dx + d_43 * dv_61;
  DataVector& dv_63 = temps.at(51);
  dv_63 = d_27 * dv_46;
  DataVector& dv_64 = temps.at(29);
  dv_64 = d_10 * dv_61;
  DataVector& dv_65 = temps.at(52);
  dv_65 = Dz * zpddot;
  DataVector& dv_66 = temps.at(1);
  dv_66 = dv_0 + dv_1;
  DataVector& dv_67 = temps.at(0);
  dv_67 = Dy + d_14 * dv_66;
  DataVector& dv_68 = temps.at(6);
  dv_68 = d_14 * (dv_7 + dv_8) + dv_56;
  DataVector& dv_69 = temps.at(53);
  dv_69 = Dx * xpddot;
  DataVector& dv_70 = temps.at(54);
  dv_70 = Dy * ypddot;
  DataVector& dv_71 = temps.at(55);
  dv_71 = dv_65 + dv_69 + dv_70;
  DataVector& dv_72 = temps.at(7);
  dv_72 = -4 * Dx * M * d_10 * d_19 * d_24 * d_3 * dv_47 -
          4 * Dx * M * d_19 * d_24 * d_36 * d_73 * dv_50 -
          4 * Dy * Dz * M * d_1 * d_24 * d_36 * d_75 * zp -
          4 * Dy * Dz * M * d_24 * d_3 * d_50 * d_59 * rp -
          4 * M * d_10 * d_115 * d_19 * d_24 * d_73 * dv_27 -
          4 * M * d_115 * d_19 * d_24 * d_50 * d_75 * dv_25 -
          4 * M * d_24 * d_59 * dv_28 * zp -
          2 * d_107 * dv_57 *
              (Dz * d_60 + d_10 * d_124 * d_73 * dv_14 + d_109 * dv_70 -
               d_11 * d_69 * dv_68 - d_110 * dv_13 - d_112 * dv_14 -
               d_114 * dv_10 + d_122 * d_27 * dv_67 + d_123 * d_59 * dv_13 +
               d_124 * d_50 * d_75 * dv_10 +
               d_34 * (d_122 * dv_4 + d_122 * dv_5 +
                       d_14 * (Dx * zpddot + Dz * xpddot) +
                       d_51 * (Dx * ypddot + Dy * xpddot) + d_74 * dv_7 +
                       d_74 * dv_8) +
               d_37 * d_73 * dv_68 +
               d_39 * d_51 *
                   (d_13 * d_59 * dv_66 + zp * (Dy * zpddot + Dz * ypddot)) +
               d_42 * dv_65 + d_44 * dv_69 - d_51 * d_69 * dv_67 -
               d_67 * dv_11 + dv_71) +
          d_110 * dv_28 + d_112 * dv_27 + d_114 * dv_25 +
          d_125 * dv_58 * 1.0 / (d_57 * d_57) + d_66 * d_94 * dv_11 * dv_49 +
          4 * d_68 * dv_23 * dv_50 +
          d_77 * (d_34 * dv_50 + dv_60 * (d_119 + d_34 * d_53) + dv_62) +
          d_78 * (Dy + d_108 * dv_63 + d_50 * dv_64 + d_51 * dv_12 +
                  dv_60 * (d_113 * d_120 + d_121 * d_51 + ypdot)) +
          d_79 * (Dz * d_42 + Dz + d_47 * dv_23 + d_47 * dv_49 +
                  dv_60 * (d_117 + d_47 * (d_116 + d_52)));
  DataVector& dv_73 = temps.at(4);
  dv_73 = (1.0 / 2.0) * 1.0 / (dv_31 * dv_31);
  DataVector& dv_74 = temps.at(14);
  dv_74 = sqrt(dv_31);
  DataVector& dv_75 = temps.at(8);
  dv_75 = dv_36 * dv_74;
  DataVector& dv_76 = temps.at(9);
  dv_76 = -d_33 * dv_18 * (d_119 + d_128 * d_34) - d_34 * dv_22 + dv_62;
  DataVector& dv_77 = temps.at(26);
  dv_77 = 2 * dv_39;
  DataVector& dv_78 = temps.at(0);
  dv_78 = dv_74 * dv_77;
  DataVector& dv_79 = temps.at(14);
  dv_79 = 1.0 / dv_74;
  DataVector& dv_80 = temps.at(54);
  dv_80 = 3 * dv_40;
  DataVector& dv_81 = temps.at(20);
  dv_81 = d_33 * dv_73;
  DataVector& dv_82 = temps.at(18);
  dv_82 = -Dy - d_35 * dv_63 + d_5 * dv_64 +
          d_58 * dv_18 * (d_120 * d_35 * d_36 - d_121 * d_6 + ypdot) +
          d_6 * dv_12;
  DataVector& dv_83 = temps.at(21);
  dv_83 = Dz * d_21 * d_25 * d_94 + Dz - d_131 * dv_21 + d_131 * dv_23 -
          d_33 * (d_117 + d_47 * (d_116 + d_7)) *
              (-d_130 * (Dy * d_16 + dv_2) +
               d_34 * (Dx * d_128 - dv_22 * xpdot) + dv_17);

  get(get<CurvedScalarWave::Tags::Psi>(*result)) =
      -d_49 * dv_40 / pow(dv_31, 3.0 / 2.0);
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(*result)) =
      -dv_73 *
      (-d_106 * dv_54 * dv_55 * 1.0 / d_76 +
       d_76 * dv_39 * dv_54 *
           (d_125 * dv_59 - d_57 * dv_72 -
            2 * (d_105 * dv_51 + dv_15) *
                (d_105 * dv_48 - d_28 - d_29 - d_30 - d_54 * dv_53 +
                 dv_52 * (d_103 * d_73 + d_104 * d_75 + d_74 * zpdot + d_96 +
                          d_97 + d_98) +
                 dv_71)) -
       d_76 * dv_42 * dv_54 *
           (Dy * dt_fy + Dz * dt_fz - d_27 * d_48 * dv_48 +
            d_27 * ft *
                (-d_12 - d_15 - d_7 - d_71 * dv_46 - d_72 * dv_44 + dv_45) -
            d_36 * d_71 * dv_33 * fy + d_47 * dv_32 * dt_fz +
            d_47 * dv_48 * fz + d_48 * dv_43 - d_6 * dv_33 * dt_fy +
            d_60 * dv_37 - d_67 * dv_37 * zp + dv_33 * dt_ft + dv_38 * dt_fx -
            dv_43 * ft -
            fx * (d_11 * dv_53 - d_34 * dv_48 - d_36 * d_73 * dv_52 + xpdot) -
            fy * ypdot - fz * zpdot) -
       3.0 / 2.0 * d_76 * dv_55 * dv_72 / sqrt(dv_59)) /
      pow(d_57, 3.0 / 2.0);
  get<0>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_81 * (dv_75 * (d_34 * (d_126 + d_127) + fx * (d_44 + 1)) -
                dv_76 * dv_79 * dv_80 +
                dv_78 * (d_32 * dv_76 + dv_34 * (d_18 * d_34 + xpdot)));
  get<1>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_81 * (3 * dv_36 * dv_39 * dv_79 * dv_82 -
                dv_75 * (d_130 * (d_127 + d_129) - fy * (d_38 + 1)) -
                dv_78 * (d_32 * dv_82 + dv_34 * (d_130 * d_18 - ypdot)));
  get<2>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -d_49 *
      (dv_31 * dv_36 * (d_47 * (d_126 + d_129 + ft) + fz * (d_42 + 1)) +
       dv_31 * dv_77 * (d_32 * dv_83 + dv_34 * (d_18 * d_47 + zpdot)) -
       dv_80 * dv_83) /
      pow(dv_31, 5.0 / 2.0);
}
}  // namespace CurvedScalarWave::Worldtube
