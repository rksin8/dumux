// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/*!
 * \file
 * \ingroup Components
 * \brief Properties of mesitylene.
 *
 */
#ifndef DUMUX_MESITYLENE_HH
#define DUMUX_MESITYLENE_HH

#include <dumux/material/idealgas.hh>
#include <dumux/material/components/component.hh>
#include <dumux/material/constants.hh>

namespace Dumux
{
/*!
 * \ingroup Components
 * \brief mesitylene
 *
 * \tparam Scalar The type used for scalar values
 */
template <class Scalar>
class Mesitylene : public Component<Scalar, Mesitylene<Scalar> >
{
    using Consts = Constants<Scalar>;

public:
    /*!
     * \brief A human readable name for the mesitylene
     */
    static std::string name()
    { return "mesitylene"; }

    /*!
     * \brief The molar mass in \f$\mathrm{[kg/mol]}\f$ of mesitylene
     */
    constexpr static Scalar molarMass()
    { return 0.120; }

    /*!
     * \brief Returns the critical temperature \f$\mathrm{[K]}\f$ of mesitylene
     */
    constexpr static Scalar criticalTemperature()
    { return 637.3; }

    /*!
     * \brief Returns the critical pressure \f$\mathrm{[Pa]}\f$ of mesitylene
     */
    constexpr static Scalar criticalPressure()
    { return 31.3e5; }

    /*!
     * \brief Returns the temperature \f$\mathrm{[K]}\f$ at mesitylene's boiling point (1 atm).
     */
    constexpr static Scalar boilingTemperature()
    { return 437.9; }

    /*!
     * \brief Returns the temperature \f$\mathrm{[K]}\f$ at mesitylene's triple point.
     */
    static Scalar tripleTemperature()
    {
        DUNE_THROW(Dune::NotImplemented, "tripleTemperature for mesitylene");
    }

    /*!
     * \brief Returns the pressure \f$\mathrm{[Pa]}\f$ at mesitylene's triple point.
     */
    static Scalar triplePressure()
    {
        DUNE_THROW(Dune::NotImplemented, "triplePressure for mesitylene");
    }

    /*!
     * \brief The saturation vapor pressure in \f$\mathrm{[Pa]}\f$ of
     *        pure mesitylene at a given temperature according to
     *        Antoine after Betz 1997, see Gmehling et al 1980
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     */
    static Scalar vaporPressure(Scalar temperature)
    {
        const Scalar A = 7.07638;
        const Scalar B = 1571.005;
        const Scalar C = 209.728;

        const Scalar T = temperature - 273.15;

        using std::pow;
        return 100 * 1.334 * pow(Scalar(10.0), Scalar(A - (B / (T + C))));
    }


    /*!
     * \brief Specific enthalpy of liquid mesitylene \f$\mathrm{[J/kg]}\f$.
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     * \param pressure pressure of component in \f$\mathrm{[Pa]}\f$
     */
    static Scalar liquidEnthalpy(const Scalar temperature,
                                 const Scalar pressure)
    {
        // Gauss quadrature rule:
        // Interval: [0K; temperature (K)]
        // Gauss-Legendre-Integration with variable transformation:
        // \int_a^b f(T) dT  \approx (b-a)/2 \sum_i=1^n \alpha_i f( (b-a)/2 x_i + (a+b)/2 )
        // with: n=2, legendre -> x_i = +/- \sqrt(1/3), \apha_i=1
        // here: a=273.15K, b=actual temperature in Kelvin
        // \leadsto h(T) = \int_273.15^T c_p(T) dT
        //              \approx 0.5 (T-273.15) * (cp( 0.5(temperature-273.15)sqrt(1/3) ) + cp(0.5(temperature-273.15)(-1)sqrt(1/3))

        // Enthalpy may have arbitrary reference state, but the empirical/fitted heatCapacity function needs Kelvin as input and is
        // fit over a certain temperature range. This suggests choosing an interval of integration being in the actual fit range.
        // I.e. choosing T=273.15K  as reference point for liquid enthalpy.
        using std::sqrt;
        const Scalar sqrt1over3 = sqrt(1./3.);
        // evaluation points according to Gauss-Legendre integration
        const Scalar TEval1 = 0.5*(temperature-273.15)*        sqrt1over3 + 0.5*(273.15+temperature);
        // evaluation points according to Gauss-Legendre integration
        const Scalar TEval2 = 0.5*(temperature-273.15)* (-1)*  sqrt1over3 + 0.5*(273.15+temperature);

        const Scalar h_n = 0.5 * (temperature-273.15) * ( liquidHeatCapacity(TEval1, pressure) + liquidHeatCapacity(TEval2, pressure) );

        return h_n;
    }

    /*!
     * \brief Latent heat of vaporization for mesitylene \f$\mathrm{[J/kg]}\f$.
     *
     * source : Reid et al. (1987, Chen method (chap. 7-11, Delta H_v = Delta H_v (T) according to chap. 7-12)) \cite reid1987
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     * \param pressure pressure of component in \f$\mathrm{[Pa]}\f$
     */
    static Scalar heatVap(Scalar temperature,
                          const  Scalar pressure)
    {
        using std::min;
        using std::max;
        temperature = min(temperature, criticalTemperature()); // regularization
        temperature = max(temperature, 0.0); // regularization

        constexpr Scalar T_crit = criticalTemperature();
        constexpr Scalar Tr1 = boilingTemperature()/criticalTemperature();
        constexpr Scalar p_crit = criticalPressure();

        //        Chen method, eq. 7-11.4 (at boiling)
        using std::log;
        const Scalar DH_v_boil = Consts::R * T_crit * Tr1 * (3.978 * Tr1 - 3.958 + 1.555*log(p_crit * 1e-5 /*Pa->bar*/ ) )
                                 / (1.07 - Tr1); /* [J/mol] */

        /* Variation with temp according to Watson relation eq 7-12.1*/
        using std::pow;
        const Scalar Tr2 = temperature/criticalTemperature();
        const Scalar n = 0.375;
        const Scalar DH_vap = DH_v_boil * pow(((1.0 - Tr2)/(1.0 - Tr1)), n);

        return (DH_vap/molarMass());          // we need [J/kg]
    }


    /*!
     * \brief Specific enthalpy of mesitylene vapor \f$\mathrm{[J/kg]}\f$.
     *
     *          This relation is true on the vapor pressure curve, i.e. as long
     *          as there is a liquid phase present.
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     * \param pressure pressure of component in \f$\mathrm{[Pa]}\f$
     */
    static Scalar gasEnthalpy(Scalar temperature, Scalar pressure)
    {
        return liquidEnthalpy(temperature,pressure) + heatVap(temperature, pressure);
    }

    /*!
     * \brief The density of mesitylene at a given pressure and temperature \f$\mathrm{[kg/m^3]}\f$ .
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     * \param pressure pressure of component in \f$\mathrm{[Pa]}\f$
     */
    static Scalar gasDensity(Scalar temperature, Scalar pressure)
    {
        return IdealGas<Scalar>::density(molarMass(),
                                         temperature,
                                         pressure);
    }

    /*!
     * \brief The density of pure mesitylene at a given pressure and temperature \f$\mathrm{[kg/m^3]}\f$.
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     * \param pressure pressure of component in \f$\mathrm{[Pa]}\f$
     */
    static Scalar liquidDensity(Scalar temperature, Scalar pressure)
    {
        return molarLiquidDensity_(temperature)*molarMass(); // [kg/m^3]
    }

    /*!
     * \brief Returns true if the gas phase is assumed to be compressible
     */
    static bool gasIsCompressible()
    { return true; }

    /*!
     * \brief Returns true if the gas phase is assumed to be ideal
     */
    static bool gasIsIdeal()
    { return true; }

    /*!
     * \brief Returns true if the liquid phase is assumed to be compressible
     */
    static constexpr bool liquidIsCompressible()
    { return false; }

    /*!
     * \brief The dynamic viscosity \f$\mathrm{[Pa*s]}\f$ of mesitylene vapor
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     * \param pressure pressure of component in \f$\mathrm{[Pa]}\f$
     * \param regularize defines, if the functions is regularized or not, set to true by default
     */
    static Scalar gasViscosity(Scalar temperature, Scalar pressure, bool regularize=true)
    {
        using std::min;
        using std::max;
        temperature = min(temperature, 500.0); // regularization
        temperature = max(temperature, 250.0);

        // reduced temperature
        Scalar Tr = temperature/criticalTemperature();

        Scalar Fp0 = 1.0;
        Scalar xi = 0.00474;

        using std::pow;
        using std::exp;
        Scalar eta_xi =
            Fp0*(0.807*pow(Tr,0.618)
                 - 0.357*exp(-0.449*Tr)
                 + 0.34*exp(-4.058*Tr)
                 + 0.018);

        return eta_xi/xi/1e7; // [Pa s]
    }

    /*!
     * \brief The dynamic viscosity \f$\mathrm{[Pa*s]}\f$ of pure mesitylene.
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     * \param pressure pressure of component in \f$\mathrm{[Pa]}\f$
     */
    static Scalar liquidViscosity(Scalar temperature, Scalar pressure)
    {
        using std::min;
        using std::max;
        temperature = min(temperature, 500.0); // regularization
        temperature = max(temperature, 250.0);

        const Scalar A = -6.749;
        const Scalar B = 2010.0;

        using std::exp;
        return exp(A + B/temperature)*1e-3; // [Pa s]
    }

    /*!
     * \brief Specific heat capacity of liquid mesitylene \f$\mathrm{[J/(kg*K)]}\f$.
     *
     * source : Reid et al. (1987, Missenard group contrib. method (chap 5-7, Table 5-11, s. example 5-8)) \cite reid1987
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     * \param pressure pressure of component in \f$\mathrm{[Pa]}\f$
     *
     */
    static Scalar liquidHeatCapacity(const Scalar temperature,
                                     const Scalar pressure)
    {
        /* according Reid et al. : Missenard group contrib. method (s. example 5-8) */
        /* Mesitylen: C9H12  : 3* CH3 ; 1* C6H5 (phenyl-ring) ; -2* H (this was to much!) */
        /* linear interpolation between table values [J/(mol K)]*/
        Scalar H, CH3, C6H5;
        if(temperature<298.) {
            // extrapolation for Temperature<273 */
            H = 13.4+1.2*(temperature-273.0)/25.;       // 13.4 + 1.2 = 14.6 = H(T=298K) i.e. interpolation of table values 273<T<298
            CH3 = 40.0+1.6*(temperature-273.0)/25.;     // 40 + 1.6 = 41.6 = CH3(T=298K)
            C6H5 = 113.0+4.2*(temperature-273.0)/25.;   // 113 + 4.2 =117.2 = C6H5(T=298K)
        }
        else if((temperature>=298.0)&&(temperature<323.)){ // i.e. interpolation of table values 298<T<323
            H = 14.6+0.9*(temperature-298.0)/25.;
            CH3 = 41.6+1.9*(temperature-298.0)/25.;
            C6H5 = 117.2+6.2*(temperature-298.0)/25.;
        }
        else if((temperature>=323.0)&&(temperature<348.)){// i.e. interpolation of table values 323<T<348
            H = 15.5+1.2*(temperature-323.0)/25.;
            CH3 = 43.5+2.3*(temperature-323.0)/25.;
            C6H5 = 123.4+6.3*(temperature-323.0)/25.;
        }
        else {
            assert(temperature>=348.0);

            /* take care: extrapolation for Temperature>373 */
            H = 16.7+2.1*(temperature-348.0)/25.;          /* leads probably to underestimates    */
            CH3 = 45.8+2.5*(temperature-348.0)/25.;
            C6H5 = 129.7+6.3*(temperature-348.0)/25.;
        }

        return (C6H5 + 3*CH3 - 2*H)/molarMass(); // J/(mol K) -> J/(kg K)
    }

    /*!
     * \brief Thermal conductivity \f$\mathrm{[[W/(m*K)]}\f$ of mesitylene
     *
     * see: http://pubs.acs.org/doi/pdf/10.1021/ci000139t
     *
     * \param temperature absolute temperature in \f$\mathrm{[K]}\f$
     * \param pressure of the phase in \f$\mathrm{[Pa]}\f$
     */
    static Scalar liquidThermalConductivity( Scalar temperature,  Scalar pressure)
    {
        return 0.1351;
    }

protected:
    /*!
     * \brief The molar density of pure mesitylene at a given pressure and temperature
     * \f$\mathrm{[mol/m^3]}\f$.
     *
     * source : Reid et al. (1987, Modified Racket technique (chap. 3-11, eq. 3-11.9)) \cite reid1987
     *
     * \param temperature temperature of component in \f$\mathrm{[K]}\f$
     */
    static Scalar molarLiquidDensity_(Scalar temperature)
    {
        using std::min;
        using std::max;
        temperature = min(temperature, 500.0); // regularization
        temperature = max(temperature, 250.0);

        const Scalar Z_RA = 0.2556; // from equation

        using std::pow;
        const Scalar expo = 1.0 + pow(1.0 - temperature/criticalTemperature(), 2.0/7.0);
        Scalar V = Consts::R*criticalTemperature()/criticalPressure()*pow(Z_RA, expo); // liquid molar volume [cm^3/mol]

        return 1.0/V; // molar density [mol/m^3]
    }

};

} // end namespace

#endif
