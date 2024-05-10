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

// NOLINTNEXTLINE(google-readability-function-size, readability-function-size)
void acceleration_terms_1(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, const double ft,
    const double fx, const double fy, const double dt_ft, const double dt_fx,
    const double dt_fy, const double Du_ft, const double Du_fx,
    const double Du_fy, const double dt_Du_ft, const double dt_Du_fx,
    const double dt_Du_fy, const double bh_mass) {
  const size_t grid_size = get<0>(centered_coords).size();
  result->initialize(grid_size);
  const double xp = particle_position[0];
  const double yp = particle_position[1];
  const double xpdot = particle_velocity[0];
  const double ypdot = particle_velocity[1];
  const double xpddot = particle_acceleration[0];
  const double ypddot = particle_acceleration[1];
  const double rp = get(magnitude(particle_position));
  const double rpdot = (xp * xpdot + yp * ypdot) / rp;

  const auto& Dx = get<0>(centered_coords);
  const auto& Dy = get<1>(centered_coords);
  const auto& z = get<2>(centered_coords);

  const double M = bh_mass;

  DynamicBuffer<DataVector> temps(307, grid_size);

  const double d_0 = rp * rp * rp;
  const double d_1 = 2 * M;
  const double d_2 = xp * xpdot;
  const double d_3 = yp * ypdot;
  const double d_4 = d_2 + d_3;
  const double d_5 = 1.0 / d_0;
  const double d_6 = rp * rp;
  const double d_7 = 4 * rp;
  const double d_8 = d_4 * d_4;
  const double d_9 = xpdot * xpdot;
  const double d_10 = ypdot * ypdot;
  const double d_11 = d_10 + d_9;
  const double d_12 = M * d_4 * d_7 + d_0 * (d_11 - 1) + d_1 * d_6 + d_1 * d_8;
  const double d_13 = 1.0 / d_12;
  const double d_14 = d_13 * d_5;
  const double d_15 = ft * rp;
  const double d_16 = d_1 * yp;
  const double d_17 = yp * yp;
  const double d_18 = d_1 * d_17;
  const double d_19 = d_0 + d_18;
  const double d_20 = rp * rp * rp * rp;
  const double d_21 = M * ft;
  const double d_22 = 12 * d_21;
  const double d_23 = d_0 * d_12;
  const double d_24 = rp * rp * rp * rp * rp * rp * rp * rp * rp * rp;
  const double d_25 = fx * xp;
  const double d_26 = fy * yp;
  const double d_27 = -rp;
  const double d_28 = d_1 + d_27;
  const double d_29 = d_1 * (d_25 + d_26) + d_15 * d_28;
  const double d_30 = d_12 * d_12;
  const double d_31 = rp * rp * rp * rp * rp * rp * rp * rp * rp;
  const double d_32 = d_15 + d_26;
  const double d_33 = d_1 * xp;
  const double d_34 = xp * xp;
  const double d_35 = d_1 * d_34;
  const double d_36 = d_0 + d_35;
  const double d_37 = d_36 * fx;
  const double d_38 = d_32 * d_33 + d_37;
  const double d_39 = d_15 + d_25;
  const double d_40 = d_19 * fy;
  const double d_41 = d_16 * d_39 + d_40;
  const double d_42 = 2 * rp;
  const double d_43 = 2 * d_23;
  const double d_44 = -d_12;
  const double d_45 = d_44 * sqrt(d_44);
  const double d_46 = d_0 * d_20 * sqrt(rp);
  const double d_47 = d_45 * d_46;
  const double d_48 = 12 * ft;
  const double d_49 = d_29 * d_48;
  const double d_50 = rp * rp * rp * rp * rp * rp * rp;
  const double d_51 = d_12 * d_50;
  const double d_52 = 12 * d_38;
  const double d_53 = rp * rp * rp * rp * rp * rp;
  const double d_54 = d_12 * d_53;
  const double d_55 = 12 * d_41;
  const double d_56 = d_46 * rp;
  const double d_57 = Du_ft * d_56;
  const double d_58 = d_45 * d_57;
  const double d_59 = sqrt(d_44);
  const double d_60 = d_20 * sqrt(rp);
  const double d_61 = 3 * M;
  const double d_62 = d_61 * fx;
  const double d_63 = d_12 * d_20;
  const double d_64 = rp * rp * rp * rp * rp;
  const double d_65 = d_12 * d_64;
  const double d_66 = d_60 * rp;
  const double d_67 = d_13 * 1.0 / d_53;
  const double d_68 = d_46 * d_20;
  const double d_69 = d_45 * rpdot;
  const double d_70 = 1.0 / d_20;
  const double d_71 = 3 * rpdot;
  const double d_72 = M * d_71;
  const double d_73 = xp * xpddot;
  const double d_74 = yp * ypddot;
  const double d_75 = 2 * rpdot;
  const double d_76 = d_11 + d_73 + d_74;
  const double d_77 =
      -M * (-2 * d_10 - 2 * d_73 - 2 * d_74 - 2 * d_9 + rpdot) / d_6 +
      d_1 * d_4 * d_5 * (-d_75 + d_76) - d_70 * d_72 * d_8 + xpddot * xpdot +
      ypddot * ypdot;
  const double d_78 = 12 * d_77;
  const double d_79 = d_0 * xpdot;
  const double d_80 = d_33 * d_4 + d_33 * rp + d_79;
  const double d_81 = d_0 * ypdot;
  const double d_82 = d_16 * d_4 + d_16 * rp + d_81;
  const double d_83 = 1.0 / d_30;
  const double d_84 = -d_77;
  const double d_85 = d_20 * d_84;
  const double d_86 = d_45 * d_66;
  const double d_87 = ft * rpdot;
  const double d_88 = dt_ft * rp;
  const double d_89 = -d_3;
  const double d_90 = d_6 * rpdot;
  const double d_91 = M * d_87;
  const double d_92 = 4 * d_50;
  const double d_93 = d_77 * d_92;
  const double d_94 = 18 * rpdot;
  const double d_95 = 1.0 / d_59;
  const double d_96 = d_77 * d_95;
  const double d_97 = 1. / sqrt(rp);
  const double d_98 = rp * rp * rp * rp * rp * rp * rp * rp;
  const double d_99 = fy * ypdot + dt_fy * yp;
  const double d_100 = fx * xpdot + dt_fx * xp;
  const double d_101 =
      d_1 * (d_100 + d_99) + d_28 * d_88 + d_75 * ft * (M + d_27);
  const double d_102 = 4 * M;
  const double d_103 = d_6 * d_71;
  const double d_104 = d_87 + d_88;
  const double d_105 = d_1 * d_32 * xpdot + d_33 * (d_104 + d_99) +
                       d_36 * dt_fx + fx * (d_102 * d_2 + d_103);
  const double d_106 = d_1 * d_39 * ypdot + d_16 * (d_100 + d_104) +
                       d_19 * dt_fy + fy * (d_102 * d_3 + d_103);
  const double d_107 = 12 * d_58;
  const double d_108 = d_0 * sqrt(rp);
  const double d_109 = 24 * d_77;
  const double d_110 = d_29 * ft;
  const double d_111 = d_53 * d_53;
  const double d_112 = d_46 * d_0;
  const double d_113 = d_110 * d_51;
  const double d_114 = d_38 * fx;
  const double d_115 = d_41 * fy;
  const double d_116 = M * d_94;
  const double d_117 = 6 * M;
  const double d_118 = d_50 * d_77;
  const double d_119 = d_24 * d_30;
  const double d_120 = d_30 * d_31;
  const double d_121 = 14 * d_23 * rpdot;
  const double d_122 = 4 * rpdot;
  const double d_123 = 2 * yp;
  const double d_124 = d_20 * xpdot;
  const double d_125 = 7 * rpdot;
  const double d_126 = d_20 * ypdot;
  const double d_127 = M * xp;
  const double d_128 = d_127 * d_48;
  const double d_129 = 12 * d_120;
  const double d_130 = Du_fy * d_127;
  const double d_131 = Du_ft * d_66;
  const double d_132 = 4 * yp;
  const double d_133 = Du_fx * d_36;
  const double d_134 = d_52 * fx;
  const double d_135 = d_55 * fy;
  const double d_136 = d_21 * d_65;
  const double d_137 = d_21 * d_6;
  const double d_138 = d_61 * fy;
  const double d_139 = 9 * M;
  const double d_140 = d_63 * fx;
  const double d_141 = d_63 * fy;
  const double d_142 = 4 * xp;
  const double d_143 = 4 * d_23;
  const double d_144 = d_33 * yp;
  const double d_145 = d_83 / d_111;
  const double d_146 = M * yp;
  const double d_147 = d_146 * d_48;
  const double d_148 = Du_fx * d_127;
  const double d_149 = Du_fy * d_19;
  const double d_150 = 4 * ft;
  const double d_151 = 4 * d_38;
  const double d_152 = 4 * d_41;
  const double d_153 = d_17 + d_34;
  const double d_154 = d_4 + rp;
  DataVector& dv_0 = temps.at(0);
  dv_0 = Dx * xpdot;
  DataVector& dv_1 = temps.at(1);
  dv_1 = Dy * ypdot;
  DataVector& dv_2 = temps.at(2);
  dv_2 = dv_0 + dv_1;
  DataVector& dv_3 = temps.at(3);
  dv_3 = Dx * xp;
  DataVector& dv_4 = temps.at(4);
  dv_4 = Dy * yp;
  DataVector& dv_5 = temps.at(5);
  dv_5 = dv_3 + dv_4;
  DataVector& dv_6 = temps.at(6);
  dv_6 = d_1 * dv_5;
  DataVector& dv_7 = temps.at(7);
  dv_7 = dv_6 * rp;
  DataVector& dv_8 = temps.at(8);
  dv_8 = d_0 * dv_2 + d_4 * dv_6 + dv_7;
  DataVector& dv_9 = temps.at(9);
  dv_9 = dv_8 * dv_8;
  DataVector& dv_10 = temps.at(10);
  dv_10 = dv_5 * dv_5;
  DataVector& dv_11 = temps.at(11);
  dv_11 = 2 * dv_10;
  DataVector& dv_12 = temps.at(12);
  dv_12 = Dx * Dx;
  DataVector& dv_13 = temps.at(13);
  dv_13 = Dy * Dy;
  DataVector& dv_14 = temps.at(14);
  dv_14 = z * z;
  DataVector& dv_15 = temps.at(15);
  dv_15 = dv_13 + dv_14;
  DataVector& dv_16 = temps.at(16);
  dv_16 = dv_12 + dv_15;
  DataVector& dv_17 = temps.at(17);
  dv_17 = M * d_5 * dv_11 - d_14 * dv_9 + dv_16;
  DataVector& dv_18 = temps.at(18);
  dv_18 = dv_17 * sqrt(dv_17);
  DataVector& dv_19 = temps.at(19);
  dv_19 = d_15 * dv_9;
  DataVector& dv_20 = temps.at(20);
  dv_20 = M * dv_5;
  DataVector& dv_21 = temps.at(21);
  dv_21 = 12 * dv_20;
  DataVector& dv_22 = temps.at(22);
  dv_22 = Dx * d_0;
  DataVector& dv_23 = temps.at(23);
  dv_23 = dv_6 * xp;
  DataVector& dv_24 = temps.at(24);
  dv_24 = dv_22 + dv_23;
  DataVector& dv_25 = temps.at(25);
  dv_25 = dv_9 * fx;
  DataVector& dv_26 = temps.at(26);
  dv_26 = dv_24 * dv_25;
  DataVector& dv_27 = temps.at(27);
  dv_27 = 6 * dv_26;
  DataVector& dv_28 = temps.at(28);
  dv_28 = Dy * d_19 + d_16 * dv_3;
  DataVector& dv_29 = temps.at(29);
  dv_29 = dv_9 * fy;
  DataVector& dv_30 = temps.at(30);
  dv_30 = dv_28 * dv_29;
  DataVector& dv_31 = temps.at(31);
  dv_31 = 6 * dv_30;
  DataVector& dv_32 = temps.at(32);
  dv_32 = d_12 * dv_5;
  DataVector& dv_33 = temps.at(33);
  dv_33 = d_20 * dv_32;
  DataVector& dv_34 = temps.at(34);
  dv_34 = dv_17 * dv_33;
  DataVector& dv_35 = temps.at(35);
  dv_35 = dv_24 * fx;
  DataVector& dv_36 = temps.at(36);
  dv_36 = d_23 * dv_17;
  DataVector& dv_37 = temps.at(37);
  dv_37 = dv_35 * dv_36;
  DataVector& dv_38 = temps.at(38);
  dv_38 = dv_28 * fy;
  DataVector& dv_39 = temps.at(39);
  dv_39 = dv_36 * dv_38;
  DataVector& dv_40 = temps.at(40);
  dv_40 = dv_17 * dv_17 * dv_17;
  DataVector& dv_41 = temps.at(41);
  dv_41 = d_6 * dv_16 - dv_11;
  DataVector& dv_42 = temps.at(42);
  dv_42 = dv_3 * dv_4;
  DataVector& dv_43 = temps.at(43);
  dv_43 = -dv_12;
  DataVector& dv_44 = temps.at(15);
  dv_44 = dv_15 + dv_43;
  DataVector& dv_45 = temps.at(44);
  dv_45 = -dv_13;
  DataVector& dv_46 = temps.at(45);
  dv_46 = dv_12 + dv_14 + dv_45;
  DataVector& dv_47 = temps.at(46);
  dv_47 = d_17 * dv_46 + d_34 * dv_44 - 4 * dv_42;
  DataVector& dv_48 = temps.at(14);
  dv_48 = 2 * dv_14;
  DataVector& dv_49 = temps.at(43);
  dv_49 = 2 * dv_13 + dv_43 + dv_48;
  DataVector& dv_50 = temps.at(12);
  dv_50 = 2 * dv_12 + dv_45 + dv_48;
  DataVector& dv_51 = temps.at(14);
  dv_51 = -6 * Dx * Dy * xp * yp + d_17 * dv_50 + d_34 * dv_49;
  DataVector& dv_52 = temps.at(44);
  dv_52 = -dv_51;
  DataVector& dv_53 = temps.at(13);
  dv_53 = -d_4 * dv_52 + d_42 * dv_47;
  DataVector& dv_54 = temps.at(47);
  dv_54 = -dv_53;
  DataVector& dv_55 = temps.at(48);
  dv_55 = dv_17 * dv_54;
  DataVector& dv_56 = temps.at(49);
  dv_56 = dv_32 * dv_52;
  DataVector& dv_57 = temps.at(50);
  dv_57 = -dv_54 * dv_8 + dv_56;
  DataVector& dv_58 = temps.at(51);
  dv_58 = -dv_57;
  DataVector& dv_59 = temps.at(52);
  dv_59 = 3 * dv_8;
  DataVector& dv_60 = temps.at(53);
  dv_60 = d_43 * dv_55 + dv_58 * dv_59;
  DataVector& dv_61 = temps.at(54);
  dv_61 = d_0 * d_12 * dv_17 * dv_41 * dv_8 - dv_20 * dv_60;
  DataVector& dv_62 = temps.at(55);
  dv_62 = M * dv_59;
  DataVector& dv_63 = temps.at(56);
  dv_63 = dv_62 * fx;
  DataVector& dv_64 = temps.at(57);
  dv_64 = d_12 * dv_52;
  DataVector& dv_65 = temps.at(58);
  dv_65 = dv_64 * xp;
  DataVector& dv_66 = temps.at(59);
  dv_66 = d_0 * dv_17;
  DataVector& dv_67 = temps.at(60);
  dv_67 = dv_65 * dv_66;
  DataVector& dv_68 = temps.at(61);
  dv_68 = dv_67 * dv_8;
  DataVector& dv_69 = temps.at(62);
  dv_69 = dv_24 * dv_60 + dv_68;
  DataVector& dv_70 = temps.at(63);
  dv_70 = dv_69 * rp;
  DataVector& dv_71 = temps.at(64);
  dv_71 = dv_62 * fy;
  DataVector& dv_72 = temps.at(65);
  dv_72 = dv_64 * yp;
  DataVector& dv_73 = temps.at(66);
  dv_73 = dv_66 * dv_72;
  DataVector& dv_74 = temps.at(67);
  dv_74 = dv_73 * dv_8;
  DataVector& dv_75 = temps.at(68);
  dv_75 = dv_28 * dv_60 + dv_74;
  DataVector& dv_76 = temps.at(69);
  dv_76 = dv_75 * rp;
  DataVector& dv_77 = temps.at(70);
  dv_77 = Du_fx * dv_24;
  DataVector& dv_78 = temps.at(71);
  dv_78 = 6 * dv_8;
  DataVector& dv_79 = temps.at(72);
  dv_79 = dv_17 * dv_17;
  DataVector& dv_80 = temps.at(73);
  dv_80 = d_47 * dv_79;
  DataVector& dv_81 = temps.at(74);
  dv_81 = dv_78 * dv_80;
  DataVector& dv_82 = temps.at(75);
  dv_82 = Du_fy * dv_28;
  DataVector& dv_83 = temps.at(76);
  dv_83 = dv_79 * dv_9;
  DataVector& dv_84 = temps.at(77);
  dv_84 = d_51 * dv_83;
  DataVector& dv_85 = temps.at(78);
  dv_85 = d_54 * dv_79;
  DataVector& dv_86 = temps.at(79);
  dv_86 = dv_25 * dv_85;
  DataVector& dv_87 = temps.at(80);
  dv_87 = dv_29 * dv_85;
  DataVector& dv_88 = temps.at(81);
  dv_88 = dv_79 * dv_8;
  DataVector& dv_89 = temps.at(82);
  dv_89 = d_58 * dv_21;
  DataVector& dv_90 = temps.at(83);
  dv_90 = d_59 * dv_77;
  DataVector& dv_91 = temps.at(84);
  dv_91 = 2 * dv_17;
  DataVector& dv_92 = temps.at(85);
  dv_92 = dv_8 * dv_8 * dv_8;
  DataVector& dv_93 = temps.at(86);
  dv_93 = d_60 * dv_92;
  DataVector& dv_94 = temps.at(87);
  dv_94 = dv_91 * dv_93;
  DataVector& dv_95 = temps.at(88);
  dv_95 = dv_22 * dv_56;
  DataVector& dv_96 = temps.at(89);
  dv_96 = dv_54 * dv_8;
  DataVector& dv_97 = temps.at(90);
  dv_97 = dv_22 * dv_96 + dv_23 * dv_58 + dv_67 - dv_95;
  DataVector& dv_98 = temps.at(91);
  dv_98 = d_63 * dv_17;
  DataVector& dv_99 = temps.at(92);
  dv_99 = dv_97 * dv_98;
  DataVector& dv_100 = temps.at(93);
  dv_100 = Dy * d_0;
  DataVector& dv_101 = temps.at(94);
  dv_101 = dv_100 * dv_56;
  DataVector& dv_102 = temps.at(95);
  dv_102 = dv_6 * yp;
  DataVector& dv_103 = temps.at(96);
  dv_103 = dv_100 * dv_96 - dv_101 + dv_102 * dv_58 + dv_73;
  DataVector& dv_104 = temps.at(97);
  dv_104 = dv_103 * fy;
  DataVector& dv_105 = temps.at(98);
  dv_105 = d_61 * dv_98;
  DataVector& dv_106 = temps.at(42);
  dv_106 = -d_17 * dv_50 - d_34 * dv_49 + 6 * dv_42;
  DataVector& dv_107 = temps.at(99);
  dv_107 = dv_106 * dv_32;
  DataVector& dv_108 = temps.at(100);
  dv_108 = -dv_47;
  DataVector& dv_109 = temps.at(101);
  dv_109 = d_4 * dv_106 + d_42 * dv_108;
  DataVector& dv_110 = temps.at(102);
  dv_110 = -dv_107 + dv_109 * dv_8;
  DataVector& dv_111 = temps.at(103);
  dv_111 = d_65 * dv_16;
  DataVector& dv_112 = temps.at(11);
  dv_112 = d_23 * dv_11;
  DataVector& dv_113 = temps.at(104);
  dv_113 = -dv_111 * dv_17 + dv_112 * dv_17;
  DataVector& dv_114 = temps.at(105);
  dv_114 = dv_110 * dv_20 + dv_113;
  DataVector& dv_115 = temps.at(106);
  dv_115 = d_21 * dv_114;
  DataVector& dv_116 = temps.at(107);
  dv_116 = d_65 * dv_17;
  DataVector& dv_117 = temps.at(108);
  dv_117 = 6 * dv_116;
  DataVector& dv_118 = temps.at(109);
  dv_118 = Du_ft * dv_20;
  DataVector& dv_119 = temps.at(110);
  dv_119 = d_66 * dv_118;
  DataVector& dv_120 = temps.at(111);
  dv_120 = 4 * dv_17;
  DataVector& dv_121 = temps.at(87);
  dv_121 = -d_59 * dv_119 * dv_120 * dv_92 - d_59 * dv_82 * dv_94 +
           d_62 * dv_99 + dv_104 * dv_105 + dv_115 * dv_117 - dv_90 * dv_94;
  DataVector& dv_122 = temps.at(112);
  dv_122 = 6. * M * d_6 * dv_61 * dv_8 * ft +
           12. * dv_40 *
               (d_24 * d_29 * d_30 * ft + d_30 * d_31 * d_38 * fx +
                d_30 * d_31 * d_41 * fy) -
           d_49 * dv_84 - d_52 * dv_86 - d_55 * dv_87 - dv_121 - dv_63 * dv_70 -
           dv_71 * dv_76 - dv_77 * dv_81 - dv_81 * dv_82 - dv_88 * dv_89;
  DataVector& dv_123 = temps.at(113);
  dv_123 = d_1 * dv_34;
  DataVector& dv_124 = temps.at(114);
  dv_124 = dv_123 * ft + dv_19 * dv_6 + dv_26 + dv_30 + dv_37 + dv_39;
  DataVector& dv_125 = temps.at(115);
  dv_125 = dv_124 * dv_18;
  DataVector& dv_126 = temps.at(116);
  dv_126 = d_14 * dv_8;
  DataVector& dv_127 = temps.at(117);
  dv_127 = Dx + d_5 * dv_23 - d_80 * dv_126;
  DataVector& dv_128 = temps.at(116);
  dv_128 = Dy + d_5 * dv_102 - d_82 * dv_126;
  DataVector& dv_129 = temps.at(118);
  dv_129 = Dx * xpddot + Dy * ypddot;
  DataVector& dv_130 = temps.at(119);
  dv_130 = d_1 * dv_2;
  DataVector& dv_131 = temps.at(120);
  dv_131 = d_6 * dv_2;
  DataVector& dv_132 = temps.at(121);
  dv_132 = d_71 * dv_131 + d_76 * dv_6 + dv_6 * rpdot;
  DataVector& dv_133 = temps.at(122);
  dv_133 = dv_8 * rp;
  DataVector& dv_134 = temps.at(119);
  dv_134 =
      -d_70 * (2. * M * dv_2 * dv_5 * rp -
               d_13 * dv_133 *
                   (d_0 * dv_129 + d_4 * dv_130 + dv_130 * rp + dv_132) +
               3. * d_13 * dv_9 * rpdot - d_72 * dv_10 - d_83 * d_85 * dv_9) +
      dv_127 * xpdot + dv_128 * ypdot;
  DataVector& dv_135 = temps.at(123);
  dv_135 = sqrt(dv_17);
  DataVector& dv_136 = temps.at(124);
  dv_136 = 18 * dv_135;
  DataVector& dv_137 = temps.at(125);
  dv_137 = dv_6 * dv_9;
  DataVector& dv_138 = temps.at(126);
  dv_138 = dv_24 * dt_fx;
  DataVector& dv_139 = temps.at(127);
  dv_139 = -d_2 + d_89 + dv_2;
  DataVector& dv_140 = temps.at(128);
  dv_140 = d_1 * dv_139;
  DataVector& dv_141 = temps.at(129);
  dv_141 = dv_28 * dt_fy;
  DataVector& dv_142 = temps.at(130);
  dv_142 = Dx * d_6;
  DataVector& dv_143 = temps.at(131);
  dv_143 =
      d_1 * (-d_34 * xpdot + dv_4 * xpdot + xp * (d_89 + 2 * dv_0 + dv_1)) +
      d_71 * dv_142 - d_79;
  DataVector& dv_144 = temps.at(132);
  dv_144 = 3 * Dy;
  DataVector& dv_145 = temps.at(133);
  dv_145 = Dx * ypdot;
  DataVector& dv_146 = temps.at(1);
  dv_146 = 2 * dv_1;
  DataVector& dv_147 = temps.at(134);
  dv_147 = d_16 * (d_89 + dv_0 + dv_146) + d_33 * (dv_145 - xpdot * yp) - d_81 +
           d_90 * dv_144;
  DataVector& dv_148 = temps.at(118);
  dv_148 = -d_10 - d_9 + dv_129;
  DataVector& dv_149 = temps.at(121);
  dv_149 = d_4 * dv_140 + dv_132 + dv_140 * rp;
  DataVector& dv_150 = temps.at(135);
  dv_150 = d_0 * dv_148 + dv_149;
  DataVector& dv_151 = temps.at(136);
  dv_151 = dv_150 * dv_8;
  DataVector& dv_152 = temps.at(137);
  dv_152 = 4 * dv_20;
  DataVector& dv_153 = temps.at(138);
  dv_153 = 2 * dv_8;
  DataVector& dv_154 = temps.at(139);
  dv_154 = dv_150 * dv_153;
  DataVector& dv_155 = temps.at(59);
  dv_155 = dv_32 * dv_66;
  DataVector& dv_156 = temps.at(140);
  dv_156 = 6 * d_12 * d_90 * dv_17;
  DataVector& dv_157 = temps.at(141);
  dv_157 = dv_36 * fx;
  DataVector& dv_158 = temps.at(142);
  dv_158 = dv_36 * fy;
  DataVector& dv_159 = temps.at(143);
  dv_159 = dv_17 * ft;
  DataVector& dv_160 = temps.at(144);
  dv_160 = d_77 * dv_91;
  DataVector& dv_161 = temps.at(145);
  dv_161 = d_53 * dv_160;
  DataVector& dv_162 = temps.at(146);
  dv_162 = 6 * dv_18;
  DataVector& dv_163 = temps.at(147);
  dv_163 = d_59 * dv_135;
  DataVector& dv_164 = temps.at(148);
  dv_164 = 1.0 / dv_135;
  DataVector& dv_165 = temps.at(149);
  dv_165 = d_59 * dv_134;
  DataVector& dv_166 = temps.at(150);
  dv_166 = d_69 * dv_88;
  DataVector& dv_167 = temps.at(151);
  dv_167 = M * dv_139;
  DataVector& dv_168 = temps.at(152);
  dv_168 = 72 * dv_166 * d_60 * rp * rp;
  DataVector& dv_169 = temps.at(153);
  dv_169 = d_54 * dv_83;
  DataVector& dv_170 = temps.at(154);
  dv_170 = d_38 * dv_25;
  DataVector& dv_171 = temps.at(155);
  dv_171 = d_65 * dv_79;
  DataVector& dv_172 = temps.at(156);
  dv_172 = 108 * dv_171 * rpdot;
  DataVector& dv_173 = temps.at(157);
  dv_173 = d_41 * dv_29;
  DataVector& dv_174 = temps.at(76);
  dv_174 = d_110 * dv_83;
  DataVector& dv_175 = temps.at(158);
  dv_175 = d_109 * d_31 * dv_79;
  DataVector& dv_176 = temps.at(159);
  dv_176 = 6 * dv_150 * dv_80;
  DataVector& dv_177 = temps.at(144);
  dv_177 = d_46 * d_95 * dv_160 * dv_92;
  DataVector& dv_178 = temps.at(160);
  dv_178 = 24 * dv_151;
  DataVector& dv_179 = temps.at(78);
  dv_179 = dv_178 * dv_85;
  DataVector& dv_180 = temps.at(161);
  dv_180 = dv_77 * dv_93;
  DataVector& dv_181 = temps.at(162);
  dv_181 = 4 * dv_165;
  DataVector& dv_182 = temps.at(163);
  dv_182 = dv_82 * dv_93;
  DataVector& dv_183 = temps.at(164);
  dv_183 = dv_17 * fx;
  DataVector& dv_184 = temps.at(165);
  dv_184 = 72 * dv_134;
  DataVector& dv_185 = temps.at(166);
  dv_185 = d_119 * dv_79;
  DataVector& dv_186 = temps.at(167);
  dv_186 = d_120 * dv_79;
  DataVector& dv_187 = temps.at(168);
  dv_187 = dv_184 * dv_186;
  DataVector& dv_188 = temps.at(169);
  dv_188 = dv_150 * rp;
  DataVector& dv_189 = temps.at(170);
  dv_189 = d_61 * dv_188;
  DataVector& dv_190 = temps.at(62);
  dv_190 = dv_69 * fx;
  DataVector& dv_191 = temps.at(68);
  dv_191 = dv_75 * fy;
  DataVector& dv_192 = temps.at(171);
  dv_192 = dv_10 * dv_17;
  DataVector& dv_193 = temps.at(172);
  dv_193 = dv_16 * rpdot;
  DataVector& dv_194 = temps.at(173);
  dv_194 = dv_139 * rp;
  DataVector& dv_195 = temps.at(174);
  dv_195 = -dv_134;
  DataVector& dv_196 = temps.at(1);
  dv_196 = dv_0 - dv_146;
  DataVector& dv_197 = temps.at(132);
  dv_197 = dv_0 * (d_123 + dv_144);
  DataVector& dv_198 = temps.at(12);
  dv_198 = dv_4 + dv_50;
  DataVector& dv_199 = temps.at(175);
  dv_199 = Dy - yp;
  DataVector& dv_200 = temps.at(176);
  dv_200 = -dv_145 * dv_199;
  DataVector& dv_201 = temps.at(177);
  dv_201 = 3 * dv_4;
  DataVector& dv_202 = temps.at(43);
  dv_202 = dv_201 + dv_49;
  DataVector& dv_203 = temps.at(178);
  dv_203 = -d_34 * dv_196 - xp * (3 * dv_200 + dv_202 * xpdot) +
           yp * (dv_197 - dv_198 * ypdot);
  DataVector& dv_204 = temps.at(179);
  dv_204 = d_42 * dv_32;
  DataVector& dv_205 = temps.at(180);
  dv_205 = 2 * dv_5;
  DataVector& dv_206 = temps.at(181);
  dv_206 = 2 * Dy;
  DataVector& dv_207 = temps.at(182);
  dv_207 = 2 * dv_4;
  DataVector& dv_208 = temps.at(15);
  dv_208 = -d_34 * (Dy * ypdot - dv_0) +
           xp * (2 * dv_200 + xpdot * (dv_207 + dv_44)) +
           yp * (-dv_0 * (dv_206 + yp) + ypdot * (dv_4 + dv_46));
  DataVector& dv_209 = temps.at(0);
  dv_209 = Dx * d_20;
  DataVector& dv_210 = temps.at(176);
  dv_210 = dv_139 * dv_64;
  DataVector& dv_211 = temps.at(1);
  dv_211 = -d_34 * dv_196 - xp * (-3 * dv_145 * dv_199 + dv_202 * xpdot) +
           yp * (dv_197 - dv_198 * ypdot);
  DataVector& dv_212 = temps.at(175);
  dv_212 = 2 * dv_32;
  DataVector& dv_213 = temps.at(43);
  dv_213 = d_77 * dv_52;
  DataVector& dv_214 = temps.at(132);
  dv_214 = dv_205 * dv_213;
  DataVector& dv_215 = temps.at(12);
  dv_215 = d_20 * dv_96;
  DataVector& dv_216 = temps.at(89);
  dv_216 = 4 * dv_96 * rpdot;
  DataVector& dv_217 = temps.at(133);
  dv_217 = dv_17 * dv_64;
  DataVector& dv_218 = temps.at(4);
  dv_218 = d_124 * dv_217;
  DataVector& dv_219 = temps.at(45);
  dv_219 = dv_150 * dv_54;
  DataVector& dv_220 = temps.at(183);
  dv_220 = d_63 * dv_91;
  DataVector& dv_221 = temps.at(184);
  dv_221 = dv_211 * xp;
  DataVector& dv_222 = temps.at(185);
  dv_222 = d_50 * dv_213;
  DataVector& dv_223 = temps.at(7);
  dv_223 = dv_58 * dv_7;
  DataVector& dv_224 = temps.at(186);
  dv_224 = dv_194 * dv_58;
  DataVector& dv_225 = temps.at(46);
  dv_225 = 2 * d_4 * dv_211 - d_7 * dv_208 - d_75 * dv_47 + d_76 * dv_52;
  DataVector& dv_226 = temps.at(187);
  dv_226 = dv_225 * dv_8;
  DataVector& dv_227 = temps.at(188);
  dv_227 = d_20 * dv_134;
  DataVector& dv_228 = temps.at(189);
  dv_228 = -d_122 * dv_56 - d_20 * dv_214 + dv_150 * dv_54 * rp -
           dv_194 * dv_64 - dv_204 * dv_211 + dv_225 * dv_8 * rp +
           dv_54 * dv_8 * rpdot;
  DataVector& dv_229 = temps.at(190);
  dv_229 = Dy * d_20;
  DataVector& dv_230 = temps.at(133);
  dv_230 = d_126 * dv_217;
  DataVector& dv_231 = temps.at(191);
  dv_231 = d_123 * dv_17;
  DataVector& dv_232 = temps.at(192);
  dv_232 = d_63 * dv_211 * dv_231;
  DataVector& dv_233 = temps.at(191);
  dv_233 = dv_222 * dv_231;
  DataVector& dv_234 = temps.at(57);
  dv_234 = d_123 * dv_64;
  DataVector& dv_235 = temps.at(193);
  dv_235 = d_23 * dv_41;
  DataVector& dv_236 = temps.at(194);
  dv_236 = dv_235 * dv_8;
  DataVector& dv_237 = temps.at(195);
  dv_237 = dv_153 * dv_17;
  DataVector& dv_238 = temps.at(196);
  dv_238 = d_63 * dv_237;
  DataVector& dv_239 = temps.at(195);
  dv_239 = d_50 * dv_237;
  DataVector& dv_240 = temps.at(197);
  dv_240 = dv_150 * dv_17;
  DataVector& dv_241 = temps.at(198);
  dv_241 = d_63 * dv_41;
  DataVector& dv_242 = temps.at(199);
  dv_242 = dv_60 * rp;
  DataVector& dv_243 = temps.at(200);
  dv_243 = 3 * dv_58;
  DataVector& dv_244 = temps.at(48);
  dv_244 = 4 * d_12 * d_20 * dv_134 * dv_54 - d_121 * dv_55 - d_93 * dv_55 -
           dv_188 * dv_243 - dv_220 * dv_225 - dv_228 * dv_59;
  DataVector& dv_245 = temps.at(169);
  dv_245 = d_20 * dv_240;
  DataVector& dv_246 = temps.at(46);
  dv_246 = (1.0 / 12.0) * 1.0 / dv_40;
  DataVector& dv_247 = temps.at(201);
  dv_247 = square(dv_17) * sqrt(dv_17);
  DataVector& dv_248 = temps.at(202);
  dv_248 = d_119 * dv_247;
  DataVector& dv_249 = temps.at(203);
  dv_249 = d_127 * dv_247;
  DataVector& dv_250 = temps.at(204);
  dv_250 = 6 * dv_247;
  DataVector& dv_251 = temps.at(205);
  dv_251 = d_120 * dv_250;
  DataVector& dv_252 = temps.at(206);
  dv_252 = d_80 * dv_247;
  DataVector& dv_253 = temps.at(204);
  dv_253 = d_47 * dv_250;
  DataVector& dv_254 = temps.at(207);
  dv_254 = d_80 * dv_253;
  DataVector& dv_255 = temps.at(208);
  dv_255 = d_51 * dv_9;
  DataVector& dv_256 = temps.at(209);
  dv_256 = dv_18 * dv_255;
  DataVector& dv_257 = temps.at(210);
  dv_257 = d_107 * dv_8;
  DataVector& dv_258 = temps.at(211);
  dv_258 = d_54 * dv_18;
  DataVector& dv_259 = temps.at(212);
  dv_259 = 12 * dv_258 * dv_9;
  DataVector& dv_260 = temps.at(213);
  dv_260 = d_47 * dv_247;
  DataVector& dv_261 = temps.at(214);
  dv_261 = 12 * dv_8;
  DataVector& dv_262 = temps.at(215);
  dv_262 = dv_260 * dv_261 * yp;
  DataVector& dv_263 = temps.at(216);
  dv_263 = d_59 * dv_18;
  DataVector& dv_264 = temps.at(217);
  dv_264 = d_131 * dv_92;
  DataVector& dv_265 = temps.at(218);
  dv_265 = 4 * dv_263 * dv_264;
  DataVector& dv_266 = temps.at(86);
  dv_266 = dv_263 * dv_93;
  DataVector& dv_267 = temps.at(219);
  dv_267 = d_132 * dv_266;
  DataVector& dv_268 = temps.at(220);
  dv_268 = dv_162 * dv_9;
  DataVector& dv_269 = temps.at(221);
  dv_269 = d_54 * dv_268;
  DataVector& dv_270 = temps.at(213);
  dv_270 = dv_260 * dv_78;
  DataVector& dv_271 = temps.at(86);
  dv_271 = 2 * dv_266;
  DataVector& dv_272 = temps.at(222);
  dv_272 = 24 * dv_8;
  DataVector& dv_273 = temps.at(223);
  dv_273 = d_50 * dv_32;
  DataVector& dv_274 = temps.at(224);
  dv_274 = d_21 * dv_18 * dv_272 * dv_273;
  DataVector& dv_275 = temps.at(216);
  dv_275 = d_131 * dv_21 * dv_263 * dv_9;
  DataVector& dv_276 = temps.at(214);
  dv_276 = dv_258 * dv_261;
  DataVector& dv_277 = temps.at(225);
  dv_277 = d_80 * dv_276;
  DataVector& dv_278 = temps.at(226);
  dv_278 = dv_252 * dv_272;
  DataVector& dv_279 = temps.at(83);
  dv_279 = d_60 * dv_90;
  DataVector& dv_280 = temps.at(227);
  dv_280 = d_80 * dv_268;
  DataVector& dv_281 = temps.at(228);
  dv_281 = d_59 * d_60 * dv_82;
  DataVector& dv_282 = temps.at(229);
  dv_282 = d_54 * dv_278;
  DataVector& dv_283 = temps.at(230);
  dv_283 = dv_127 * dv_18;
  DataVector& dv_284 = temps.at(231);
  dv_284 = d_119 * d_48 * dv_20;
  DataVector& dv_285 = temps.at(232);
  dv_285 = d_120 * dv_127;
  DataVector& dv_286 = temps.at(233);
  dv_286 = dv_162 * dv_285;
  DataVector& dv_287 = temps.at(234);
  dv_287 = d_49 * dv_248;
  DataVector& dv_288 = temps.at(232);
  dv_288 = dv_247 * dv_285;
  DataVector& dv_289 = temps.at(235);
  dv_289 = dv_127 * dv_135;
  DataVector& dv_290 = temps.at(223);
  dv_290 = dv_273 * dv_9;
  DataVector& dv_291 = temps.at(236);
  dv_291 = 36 * d_21 * dv_290;
  DataVector& dv_292 = temps.at(237);
  dv_292 = dv_8 * dv_89;
  DataVector& dv_293 = temps.at(217);
  dv_293 = dv_163 * dv_21 * dv_264;
  DataVector& dv_294 = temps.at(238);
  dv_294 = d_54 * dv_127;
  DataVector& dv_295 = temps.at(239);
  dv_295 = dv_136 * dv_294;
  DataVector& dv_296 = temps.at(240);
  dv_296 = d_47 * dv_77;
  DataVector& dv_297 = temps.at(241);
  dv_297 = dv_162 * dv_8;
  DataVector& dv_298 = temps.at(242);
  dv_298 = dv_127 * dv_297;
  DataVector& dv_299 = temps.at(243);
  dv_299 = 6 * dv_163;
  DataVector& dv_300 = temps.at(244);
  dv_300 = dv_127 * dv_299;
  DataVector& dv_301 = temps.at(245);
  dv_301 = d_47 * dv_82;
  DataVector& dv_302 = temps.at(238);
  dv_302 = dv_18 * dv_294;
  DataVector& dv_303 = temps.at(246);
  dv_303 = d_52 * dv_25;
  DataVector& dv_304 = temps.at(247);
  dv_304 = d_55 * dv_29;
  DataVector& dv_305 = temps.at(104);
  dv_305 = d_136 * dv_136 * (dv_113 + dv_20 * dv_58);
  DataVector& dv_306 = temps.at(248);
  dv_306 = d_80 * dv_135;
  DataVector& dv_307 = temps.at(249);
  dv_307 = d_137 * dv_61;
  DataVector& dv_308 = temps.at(250);
  dv_308 = 6 * dv_307;
  DataVector& dv_309 = temps.at(251);
  dv_309 = d_62 * dv_70;
  DataVector& dv_310 = temps.at(252);
  dv_310 = d_138 * dv_76;
  DataVector& dv_311 = temps.at(253);
  dv_311 = d_139 * dv_289;
  DataVector& dv_312 = temps.at(254);
  dv_312 = dv_22 * dv_53;
  DataVector& dv_313 = temps.at(255);
  dv_313 = dv_36 * dv_51;
  DataVector& dv_314 = temps.at(256);
  dv_314 = d_140 * (Dx * d_0 * d_12 * dv_5 * dv_51 - dv_23 * dv_57 -
                    dv_312 * dv_8 - dv_313 * xp);
  DataVector& dv_315 = temps.at(13);
  dv_315 = d_0 * dv_53;
  DataVector& dv_316 = temps.at(257);
  dv_316 = dv_315 * dv_8;
  DataVector& dv_317 = temps.at(255);
  dv_317 = d_141 * (Dy * d_0 * d_12 * dv_5 * dv_51 - Dy * dv_316 -
                    dv_102 * dv_57 - dv_313 * yp);
  DataVector& dv_318 = temps.at(258);
  dv_318 = dv_127 * dv_164;
  DataVector& dv_319 = temps.at(249);
  dv_319 = dv_307 * dv_8;
  DataVector& dv_320 = temps.at(259);
  dv_320 = 30 * dv_319;
  DataVector& dv_321 = temps.at(260);
  dv_321 = M * dv_133;
  DataVector& dv_322 = temps.at(261);
  dv_322 = 15 * dv_321;
  DataVector& dv_323 = temps.at(262);
  dv_323 = dv_318 * dv_322;
  DataVector& dv_324 = temps.at(263);
  dv_324 = d_143 * dv_127;
  DataVector& dv_325 = temps.at(264);
  dv_325 = 2 * dv_111;
  DataVector& dv_326 = temps.at(265);
  dv_326 = Dx * d_34;
  DataVector& dv_327 = temps.at(266);
  dv_327 = Dx * d_17;
  DataVector& dv_328 = temps.at(177);
  dv_328 = dv_201 * xp + dv_326 - 2 * dv_327;
  DataVector& dv_329 = temps.at(267);
  dv_329 = dv_212 * dv_328;
  DataVector& dv_330 = temps.at(182);
  dv_330 = d_4 * dv_328 + d_42 * (dv_207 * xp + dv_326 - dv_327);
  DataVector& dv_331 = temps.at(266);
  dv_331 = -d_80 * dv_54 + dv_329 - 2 * dv_330 * dv_8 + dv_65;
  DataVector& dv_332 = temps.at(265);
  dv_332 = -dv_331;
  DataVector& dv_333 = temps.at(268);
  dv_333 = d_136 * dv_162;
  DataVector& dv_334 = temps.at(269);
  dv_334 = d_0 * dv_206 * dv_32;
  DataVector& dv_335 = temps.at(13);
  dv_335 = Dy * dv_315;
  DataVector& dv_336 = temps.at(270);
  dv_336 = d_123 * d_23 * dv_51;
  DataVector& dv_337 = temps.at(271);
  dv_337 = d_144 * dv_57;
  DataVector& dv_338 = temps.at(272);
  dv_338 = d_63 * dv_18;
  DataVector& dv_339 = temps.at(273);
  dv_339 = d_138 * dv_338;
  DataVector& dv_340 = temps.at(274);
  dv_340 = d_43 * dv_51 * xp;
  DataVector& dv_341 = temps.at(257);
  dv_341 = -d_0 * d_12 * dv_5 * dv_51 + dv_316;
  DataVector& dv_342 = temps.at(272);
  dv_342 = d_62 * dv_338;
  DataVector& dv_343 = temps.at(275);
  dv_343 = dv_153 * dv_36;
  DataVector& dv_344 = temps.at(276);
  dv_344 = 4 * dv_36;
  DataVector& dv_345 = temps.at(277);
  dv_345 = d_80 * dv_243 + dv_324 * dv_54 + dv_330 * dv_344 + dv_332 * dv_59;
  DataVector& dv_346 = temps.at(71);
  dv_346 = d_137 * dv_135 * dv_78;
  DataVector& dv_347 = temps.at(278);
  dv_347 = d_123 * dv_36;
  DataVector& dv_348 = temps.at(279);
  dv_348 = d_0 * dv_8;
  DataVector& dv_349 = temps.at(280);
  dv_349 = dv_234 * dv_348;
  DataVector& dv_350 = temps.at(281);
  dv_350 = d_144 * dv_60;
  DataVector& dv_351 = temps.at(282);
  dv_351 = dv_135 * rp;
  DataVector& dv_352 = temps.at(64);
  dv_352 = dv_351 * dv_71;
  DataVector& dv_353 = temps.at(283);
  dv_353 = dv_343 * xp;
  DataVector& dv_354 = temps.at(284);
  dv_354 = d_0 * dv_153 * dv_65;
  DataVector& dv_355 = temps.at(282);
  dv_355 = dv_351 * dv_63;
  DataVector& dv_356 = temps.at(56);
  dv_356 = d_145 * dv_246;
  DataVector& dv_357 = temps.at(285);
  dv_357 = d_146 * dv_247;
  DataVector& dv_358 = temps.at(286);
  dv_358 = d_82 * dv_247;
  DataVector& dv_359 = temps.at(204);
  dv_359 = d_82 * dv_253;
  DataVector& dv_360 = temps.at(214);
  dv_360 = d_82 * dv_276;
  DataVector& dv_361 = temps.at(222);
  dv_361 = dv_272 * dv_358;
  DataVector& dv_362 = temps.at(220);
  dv_362 = d_82 * dv_268;
  DataVector& dv_363 = temps.at(287);
  dv_363 = d_54 * dv_361;
  DataVector& dv_364 = temps.at(288);
  dv_364 = dv_128 * dv_18;
  DataVector& dv_365 = temps.at(289);
  dv_365 = d_120 * dv_128;
  DataVector& dv_366 = temps.at(290);
  dv_366 = dv_162 * dv_365;
  DataVector& dv_367 = temps.at(201);
  dv_367 = dv_247 * dv_365;
  DataVector& dv_368 = temps.at(289);
  dv_368 = dv_128 * dv_135;
  DataVector& dv_369 = temps.at(291);
  dv_369 = d_54 * dv_128 * dv_136;
  DataVector& dv_370 = temps.at(241);
  dv_370 = dv_128 * dv_297;
  DataVector& dv_371 = temps.at(243);
  dv_371 = dv_128 * dv_299;
  DataVector& dv_372 = temps.at(211);
  dv_372 = dv_128 * dv_258;
  DataVector& dv_373 = temps.at(292);
  dv_373 = d_82 * dv_135;
  DataVector& dv_374 = temps.at(293);
  dv_374 = d_139 * dv_368;
  DataVector& dv_375 = temps.at(294);
  dv_375 = dv_128 * dv_164;
  DataVector& dv_376 = temps.at(261);
  dv_376 = dv_322 * dv_375;
  DataVector& dv_377 = temps.at(295);
  dv_377 = d_143 * dv_128;
  DataVector& dv_378 = temps.at(296);
  dv_378 = Dy * d_17;
  DataVector& dv_379 = temps.at(297);
  dv_379 = -d_34 * dv_206 + 3 * dv_3 * yp + dv_378;
  DataVector& dv_380 = temps.at(298);
  dv_380 = dv_212 * dv_379;
  DataVector& dv_381 = temps.at(3);
  dv_381 = d_123 * dv_3;
  DataVector& dv_382 = temps.at(296);
  dv_382 = d_4 * dv_379 + d_42 * (-Dy * d_34 + dv_378 + dv_381);
  DataVector& dv_383 = temps.at(299);
  dv_383 = -d_82 * dv_54 + dv_380 - 2 * dv_382 * dv_8 + dv_72;
  DataVector& dv_384 = temps.at(300);
  dv_384 = -dv_383;
  DataVector& dv_385 = temps.at(200);
  dv_385 = d_82 * dv_243 + dv_344 * dv_382 + dv_377 * dv_54 + dv_384 * dv_59;
  DataVector& dv_386 = temps.at(167);
  dv_386 = 2 * dv_186;
  DataVector& dv_387 = temps.at(276);
  dv_387 = d_30 * dv_40;
  DataVector& dv_388 = temps.at(47);
  dv_388 = d_31 * dv_387;
  DataVector& dv_389 = temps.at(260);
  dv_389 = 5 * dv_321;
  DataVector& dv_390 = temps.at(301);
  dv_390 = d_54 * dv_17;
  DataVector& dv_391 = temps.at(32);
  dv_391 = d_153 * (-d_154 * dv_8 + dv_32);
  DataVector& dv_392 = temps.at(73);
  dv_392 = dv_153 * dv_80;
  DataVector& dv_393 = temps.at(52);
  dv_393 = -d_153 * d_154 * d_43 * dv_17 + d_23 * dv_109 + dv_391 * dv_59;
  DataVector& dv_394 = temps.at(302);
  dv_394 = 2 * dv_393;
  DataVector& dv_395 = temps.at(303);
  dv_395 = d_1 * dv_133;
  DataVector& dv_396 = temps.at(304);
  dv_396 = d_23 * dv_106;
  DataVector& dv_397 = temps.at(305);
  dv_397 = dv_20 * dv_391;
  DataVector& dv_398 = temps.at(306);
  dv_398 = d_1 * dv_79;

  get(get<CurvedScalarWave::Tags::Psi>(*result)) =
      -d_67 / dv_18 / 12. *
      (d_22 * dv_34 + d_67 * dv_122 / dv_17 + dv_19 * dv_21 + dv_27 + dv_31 +
       6. * dv_37 + 6. * dv_39);
  get(get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(*result)) =
      -dv_246 *
      (-d_56 * d_59 * d_78 * dv_125 + 54 * d_60 * d_69 * dv_125 -
       d_86 * dv_124 * dv_134 * dv_136 +
       d_86 * dv_162 *
           (4 * M * d_12 * d_20 * dv_134 * dv_5 * ft +
            2 * d_0 * d_12 * dv_134 * dv_24 * fx +
            2 * d_0 * d_12 * dv_134 * dv_28 * fy - d_15 * dv_151 * dv_152 -
            d_87 * dv_137 - d_88 * dv_137 - 14 * d_91 * dv_155 -
            d_93 * dv_159 * dv_20 - dv_123 * dt_ft - dv_138 * dv_36 -
            dv_138 * dv_9 - dv_140 * dv_19 - dv_140 * dv_98 * ft -
            dv_141 * dv_36 - dv_141 * dv_9 - dv_143 * dv_157 - dv_143 * dv_25 -
            dv_147 * dv_158 - dv_147 * dv_29 - dv_154 * dv_35 - dv_154 * dv_38 -
            dv_156 * dv_35 - dv_156 * dv_38 - dv_161 * dv_35 - dv_161 * dv_38) -
       d_94 * dv_122 * dv_163 / (rp * sqrt(rp)) +
       4 * d_96 * dv_122 * dv_135 * square(rp) * sqrt(rp) +
       5 * d_97 * dv_122 * dv_164 * dv_165 +
       d_97 * dv_163 *
           (48 * Du_ft * M * d_45 * d_56 * dv_134 * dv_17 * dv_5 * dv_8 +
            28 * Du_ft * M * d_59 * d_60 * dv_17 * dv_5 * dv_92 * rpdot +
            4 * Du_ft * M * d_59 * d_66 * dv_139 * dv_17 * dv_92 +
            12 * Du_ft * M * d_59 * d_66 * dv_150 * dv_17 * dv_5 * dv_9 +
            36 * Du_ft * M * d_59 * d_68 * d_77 * dv_5 * dv_79 * dv_8 +
            4 * dt_Du_ft * M * d_59 * d_66 * dv_17 * dv_5 * dv_92 -
            dt_Du_ft * d_45 * d_56 * dv_21 * dv_88 +
            12 * Du_fx * d_108 * d_59 * dv_17 * dv_24 * dv_92 * rpdot +
            18 * Du_fx * d_112 * d_59 * d_77 * dv_24 * dv_79 * dv_8 +
            24 * Du_fx * d_45 * d_46 * dv_134 * dv_17 * dv_24 * dv_8 +
            2 * Du_fx * d_59 * d_60 * dv_143 * dv_17 * dv_92 +
            6 * Du_fx * d_59 * d_60 * dv_150 * dv_17 * dv_24 * dv_9 -
            Du_fx * dv_143 * dv_81 +
            2 * dt_Du_fx * d_59 * d_60 * dv_17 * dv_24 * dv_92 -
            dt_Du_fx * dv_24 * dv_81 +
            12 * Du_fy * d_108 * d_59 * dv_17 * dv_28 * dv_92 * rpdot +
            18 * Du_fy * d_112 * d_59 * d_77 * dv_28 * dv_79 * dv_8 +
            24 * Du_fy * d_45 * d_46 * dv_134 * dv_17 * dv_28 * dv_8 +
            2 * Du_fy * d_59 * d_60 * dv_147 * dv_17 * dv_92 +
            6 * Du_fy * d_59 * d_60 * dv_150 * dv_17 * dv_28 * dv_9 -
            Du_fy * dv_147 * dv_81 +
            2 * dt_Du_fy * d_59 * d_60 * dv_17 * dv_28 * dv_92 -
            dt_Du_fy * dv_28 * dv_81 +
            6 * M * d_12 * d_20 * dv_103 * dv_134 * fy +
            6 * M * d_12 * d_20 * dv_134 * dv_97 * fx +
            12 * M * d_12 * d_64 * dv_114 * dv_134 * ft +
            6 * M * d_6 * dv_150 * dv_61 * ft +
            6 * M * d_6 * dv_61 * dv_8 * dt_ft - M * dv_114 * dv_117 * dt_ft +
            6 * M * dv_61 * dv_8 * ft * rp * rpdot +
            6 * M * dv_8 * ft * rp *
                (d_125 * dv_17 * dv_236 + d_77 * dv_239 * dv_41 -
                 dv_134 * dv_153 * dv_241 - dv_167 * dv_242 + dv_20 * dv_244 -
                 dv_238 * (dv_131 + dv_139 * dv_205 - dv_193 * rp) +
                 dv_240 * dv_241) +
            3 * M * dv_8 * fx *
                (2 * d_12 * d_20 * dv_134 * dv_52 * dv_8 * xp - d_125 * dv_68 -
                 dv_143 * dv_242 - dv_213 * dv_239 * xp - dv_218 * dv_8 -
                 dv_221 * dv_238 + dv_24 * dv_244 - dv_245 * dv_65) +
            3 * M * dv_8 * fy *
                (2 * d_12 * d_20 * dv_134 * dv_52 * dv_8 * yp - d_125 * dv_74 -
                 dv_147 * dv_242 - dv_230 * dv_8 - dv_232 * dv_8 -
                 dv_233 * dv_8 + dv_244 * dv_28 - dv_245 * dv_72) +
            12 * d_101 * d_24 * d_30 * dv_40 * ft - d_101 * d_48 * dv_84 +
            12 * d_105 * d_30 * d_31 * dv_40 * fx - 12 * d_105 * dv_86 +
            12 * d_106 * d_30 * d_31 * dv_40 * fy - 12 * d_106 * dv_87 -
            d_107 * dv_167 * dv_88 - d_109 * d_24 * dv_174 -
            d_110 * dv_184 * dv_185 +
            48 * d_111 * d_12 * d_38 * d_77 * dv_40 * fx +
            48 * d_111 * d_12 * d_41 * d_77 * dv_40 * fy -
            d_113 * dv_178 * dv_79 - d_114 * dv_179 - d_114 * dv_187 -
            d_115 * dv_179 - d_115 * dv_187 - d_116 * dv_103 * dv_158 -
            d_116 * dv_157 * dv_97 - d_117 * d_118 * dv_104 * dv_17 -
            d_117 * d_118 * dv_183 * dv_97 +
            48 * d_12 * d_29 * d_50 * dv_134 * dv_17 * dv_9 * ft +
            48 * d_12 * d_29 * d_77 * dv_40 * ft * d_24 * d_0 +
            48 * d_12 * d_38 * d_53 * dv_134 * dv_17 * dv_9 * fx +
            48 * d_12 * d_41 * d_53 * dv_134 * dv_17 * dv_9 * fy -
            6 * d_21 * dv_98 *
                (M * dv_110 * dv_194 + d_121 * dv_192 +
                 d_31 * d_84 * dv_16 * dv_91 - 2 * d_54 * dv_16 * dv_195 +
                 d_54 * dv_2 * dv_91 + 4 * d_63 * dv_10 * dv_195 -
                 d_84 * d_92 * dv_192 - 9 * dv_116 * dv_193 +
                 dv_120 * dv_139 * dv_33 +
                 dv_20 * (-d_12 * dv_106 * dv_194 - d_122 * dv_107 +
                          d_85 * dv_106 * dv_205 + dv_109 * dv_8 * rpdot +
                          dv_109 * rp * (d_0 * dv_148 + dv_149) +
                          dv_133 * (2 * d_4 * dv_203 - d_7 * dv_208 +
                                    d_75 * dv_108 + d_76 * dv_106) -
                          dv_203 * dv_204)) +
            12 * d_24 * d_29 * d_30 * dv_40 * dt_ft +
            192 * d_29 * d_30 * d_31 * dv_40 * ft * rpdot -
            120 * d_29 * d_87 * dv_169 - 12 * d_29 * dv_84 * dt_ft +
            12 * d_30 * d_31 * d_38 * dv_40 * dt_fx +
            12 * d_30 * d_31 * d_41 * dv_40 * dt_fy +
            180 * d_30 * d_38 * d_98 * dv_40 * fx * rpdot +
            180 * d_30 * d_41 * d_98 * dv_40 * fy * rpdot -
            156 * d_46 * dv_118 * dv_166 - d_52 * dv_169 * dt_fx -
            d_55 * dv_169 * dt_fy - d_57 * d_96 * dv_120 * dv_20 * dv_92 -
            d_61 * dv_157 *
                (-Dx * d_50 * dv_214 + d_124 * dv_56 + d_125 * dv_67 -
                 d_125 * dv_95 + d_33 * dv_224 - dv_209 * dv_210 -
                 dv_209 * dv_211 * dv_212 + dv_209 * dv_219 + dv_209 * dv_226 -
                 dv_215 * xpdot + dv_216 * dv_22 + dv_218 + dv_220 * dv_221 +
                 dv_222 * dv_91 * xp + dv_223 * xpdot - 2 * dv_227 * dv_65 +
                 dv_228 * dv_23) -
            d_61 * dv_158 *
                (-d_125 * dv_101 + d_125 * dv_73 + d_126 * dv_56 +
                 d_16 * dv_224 + dv_100 * dv_216 + dv_102 * dv_228 -
                 dv_206 * dv_211 * dv_33 - dv_206 * dv_222 * dv_5 -
                 dv_210 * dv_229 - dv_215 * ypdot + dv_219 * dv_229 +
                 dv_223 * ypdot + dv_226 * dv_229 - dv_227 * dv_234 + dv_230 +
                 dv_232 + dv_233) -
            d_61 * dv_99 * dt_fx - d_78 * d_98 * dv_115 * dv_17 -
            42 * d_91 * dv_114 * dv_98 - dv_103 * dv_105 * dt_fy -
            8 * dv_119 * dv_165 * dv_92 - dv_150 * dv_79 * dv_89 -
            dv_168 * dv_77 - dv_168 * dv_82 - dv_170 * dv_172 -
            dv_170 * dv_175 - dv_172 * dv_173 - dv_173 * dv_175 -
            dv_176 * dv_77 - dv_176 * dv_82 - dv_177 * dv_77 - dv_177 * dv_82 -
            dv_180 * dv_181 - dv_181 * dv_182 - dv_189 * dv_190 -
            dv_189 * dv_191 - dv_62 * dv_70 * dt_fx - dv_62 * dv_76 * dt_fy)) *
      1.0 / d_68 / (square(d_44) * sqrt(d_44));
  get<0>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_356 *
      (-d_113 * dv_278 - d_114 * dv_282 - d_115 * dv_282 +
       d_127 * d_26 * dv_259 + d_127 * dv_265 + d_128 * dv_248 +
       d_128 * dv_256 + d_129 * d_26 * dv_249 - d_130 * dv_262 +
       d_130 * dv_267 - d_133 * dv_270 + d_133 * dv_271 + d_134 * dv_288 +
       d_135 * dv_288 + d_37 * dv_251 + d_37 * dv_269 + d_49 * dv_255 * dv_283 +
       d_80 * dv_274 + d_80 * dv_275 + dv_127 * dv_287 - dv_127 * dv_293 +
       dv_127 * dv_305 - dv_180 * dv_300 - dv_182 * dv_300 + dv_190 * dv_323 +
       dv_191 * dv_323 - dv_249 * dv_257 - dv_252 * dv_89 - dv_254 * dv_77 -
       dv_254 * dv_82 - dv_26 * dv_295 + dv_277 * dv_35 + dv_277 * dv_38 +
       dv_279 * dv_280 + dv_280 * dv_281 - dv_283 * dv_284 + dv_283 * dv_292 -
       dv_286 * dv_35 - dv_286 * dv_38 - dv_289 * dv_291 - dv_295 * dv_30 +
       dv_296 * dv_298 + dv_298 * dv_301 + dv_302 * dv_303 + dv_302 * dv_304 +
       dv_306 * dv_308 - dv_306 * dv_309 - dv_306 * dv_310 + dv_311 * dv_314 +
       dv_311 * dv_317 - dv_318 * dv_320 -
       dv_333 * (-2 * Dx * dv_116 + d_127 * dv_58 + d_142 * dv_155 +
                 dv_10 * dv_324 - dv_127 * dv_325 + dv_20 * dv_332) -
       dv_339 * (Dy * d_0 * d_12 * dv_51 * xp + 2 * Dy * d_0 * dv_330 * dv_8 +
                 2 * d_0 * d_12 * dv_17 * dv_328 * yp - d_80 * dv_335 -
                 dv_102 * dv_331 - dv_127 * dv_336 - dv_328 * dv_334 - dv_337) -
       dv_342 * (Dx * d_0 * d_12 * dv_51 * xp + 2 * Dx * d_0 * dv_330 * dv_8 +
                 2 * d_0 * d_12 * dv_17 * dv_328 * xp - d_35 * dv_57 -
                 d_80 * dv_312 - dv_127 * dv_340 - dv_22 * dv_329 -
                 dv_23 * dv_331 - dv_341) +
       dv_346 *
           (-d_127 * dv_60 + d_80 * dv_17 * dv_235 + dv_127 * dv_153 * dv_235 -
            dv_20 * dv_345 + dv_343 * (dv_142 - dv_205 * xp)) -
       dv_352 * (d_80 * dv_73 + dv_127 * dv_349 + dv_28 * dv_345 +
                 dv_328 * dv_347 * dv_8 + dv_350) -
       dv_355 * (d_36 * dv_60 + d_80 * dv_67 + dv_127 * dv_354 +
                 dv_24 * dv_345 + dv_328 * dv_353));
  get<1>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -dv_356 *
      (-d_113 * dv_361 - d_114 * dv_363 - d_115 * dv_363 +
       d_129 * d_25 * dv_357 + d_134 * dv_367 + d_135 * dv_367 +
       d_146 * d_25 * dv_259 + d_146 * dv_265 + d_147 * dv_248 +
       d_147 * dv_256 - d_148 * dv_262 + d_148 * dv_267 - d_149 * dv_270 +
       d_149 * dv_271 + d_40 * dv_251 + d_40 * dv_269 + d_49 * dv_128 * dv_256 +
       d_82 * dv_274 + d_82 * dv_275 + dv_128 * dv_287 - dv_128 * dv_293 +
       dv_128 * dv_305 - dv_180 * dv_371 - dv_182 * dv_371 + dv_190 * dv_376 +
       dv_191 * dv_376 - dv_257 * dv_357 - dv_26 * dv_369 + dv_279 * dv_362 +
       dv_281 * dv_362 - dv_284 * dv_364 - dv_291 * dv_368 + dv_292 * dv_364 +
       dv_296 * dv_370 - dv_30 * dv_369 + dv_301 * dv_370 + dv_303 * dv_372 +
       dv_304 * dv_372 + dv_308 * dv_373 - dv_309 * dv_373 - dv_310 * dv_373 +
       dv_314 * dv_374 + dv_317 * dv_374 - dv_320 * dv_375 -
       dv_333 * (d_132 * dv_155 + d_146 * dv_58 + dv_10 * dv_377 -
                 dv_116 * dv_206 - dv_128 * dv_325 + dv_20 * dv_384) -
       dv_339 * (Dy * d_0 * d_12 * dv_51 * yp + 2 * Dy * d_0 * dv_382 * dv_8 +
                 2 * d_0 * d_12 * dv_17 * dv_379 * yp - d_18 * dv_57 -
                 d_82 * dv_335 - dv_102 * dv_383 - dv_128 * dv_336 -
                 dv_334 * dv_379 - dv_341) -
       dv_342 * (Dx * d_0 * d_12 * dv_51 * yp + 2 * Dx * d_0 * dv_382 * dv_8 +
                 2 * d_0 * d_12 * dv_17 * dv_379 * xp - d_82 * dv_312 -
                 dv_128 * dv_340 - dv_22 * dv_380 - dv_23 * dv_383 - dv_337) +
       dv_346 * (d_0 * d_12 * d_82 * dv_17 * dv_41 +
                 2 * d_0 * d_12 * dv_128 * dv_41 * dv_8 - d_146 * dv_60 -
                 dv_20 * dv_385 - dv_343 * (-Dy * (-2 * d_17 + d_6) + dv_381)) +
       dv_35 * dv_360 - dv_35 * dv_366 -
       dv_352 * (d_19 * dv_60 + d_82 * dv_73 + dv_128 * dv_349 +
                 dv_28 * dv_385 + dv_347 * dv_379 * dv_8) -
       dv_355 * (d_82 * dv_67 + dv_128 * dv_354 + dv_24 * dv_385 + dv_350 +
                 dv_353 * dv_379) -
       dv_358 * dv_89 - dv_359 * dv_77 - dv_359 * dv_82 + dv_360 * dv_38 -
       dv_366 * dv_38);
  get<2>(get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                           Frame::Inertial>>(*result)) =
      -1.0 / 4.0 * d_145 * z *
      (d_102 * d_6 * dv_159 * dv_8 * (dv_116 * dv_8 + dv_236 - dv_393 * dv_6) -
       d_102 * dv_171 * ft * (-dv_111 + dv_112 - dv_116 + dv_391 * dv_6) +
       d_12 * d_92 * dv_174 +
       d_140 * dv_398 *
           (2 * Dx * d_0 * d_153 * d_154 * dv_8 +
            2 * d_0 * d_12 * d_153 * dv_17 * xp - d_142 * dv_397 -
            d_153 * dv_212 * dv_22 - dv_396 * xp) +
       d_141 * dv_398 *
           (2 * Dy * d_0 * d_153 * d_154 * dv_8 +
            2 * d_0 * d_12 * d_153 * dv_17 * yp - d_132 * dv_397 -
            d_153 * dv_334 - dv_396 * yp) +
       d_150 * d_24 * d_29 * dv_387 - d_150 * dv_185 * dv_20 +
       d_151 * dv_388 * fx + d_151 * dv_86 + d_152 * dv_388 * fy +
       d_152 * dv_87 - d_22 * dv_17 * dv_290 + d_58 * dv_152 * dv_88 + dv_121 +
       dv_17 * dv_395 * fy *
           (2 * d_0 * d_12 * d_153 * dv_17 * dv_8 * yp - dv_28 * dv_394 -
            dv_348 * dv_72) +
       dv_183 * dv_395 *
           (2 * d_0 * d_12 * d_153 * dv_17 * dv_8 * xp - dv_24 * dv_394 -
            dv_348 * dv_65) +
       dv_190 * dv_389 + dv_191 * dv_389 - dv_27 * dv_390 - dv_31 * dv_390 -
       10 * dv_319 - dv_35 * dv_386 - dv_38 * dv_386 + dv_392 * dv_77 +
       dv_392 * dv_82) /
      (cube(dv_17) * sqrt(dv_17));
}
}  // namespace CurvedScalarWave::Worldtube
