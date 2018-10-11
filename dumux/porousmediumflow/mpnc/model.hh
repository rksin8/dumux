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
 * \ingroup MPNCModel
 * \brief A fully implicit model for MpNc flow using
 *        vertex centered finite volumes.
 *
 * This model implements a \f$M\f$-phase flow of a fluid mixture
 * composed of \f$N\f$ chemical species. The phases are denoted by
 * lower index \f$\alpha \in \{ 1, \dots, M \}\f$. All fluid phases
 * are mixtures of \f$N \geq M - 1\f$ chemical species which are
 * denoted by the upper index \f$\kappa \in \{ 1, \dots, N \} \f$.
 *
 * The momentum approximation can be selected via "BaseFluxVariables":
 * Darcy (ImplicitDarcyFluxVariables) and Forchheimer (ImplicitForchheimerFluxVariables)
 * relations are available for all Box models.
 *
 * By inserting this into the equations for the conservation of the
 * mass of each component, one gets one mass-continuity equation for
 * each component \f$\kappa\f$
 * \f[
 \sum_{\kappa} \left(
    \phi \frac{\partial \left(\varrho_\alpha x_\alpha^\kappa S_\alpha\right)}{\partial t}
    +
    \mathrm{div}\;
    \left\{
        v_\alpha
       \frac{\varrho_\alpha}{\overline M_\alpha} x_\alpha^\kappa
    \right\}
    \right)
    = q^\kappa
    \f]
 * with \f$\overline M_\alpha\f$ being the average molar mass of the
 * phase \f$\alpha\f$: \f[ \overline M_\alpha = \sum_\kappa M^\kappa
 * \; x_\alpha^\kappa \f]
 *
 * For the missing \f$M\f$ model assumptions, the model assumes that
 * if a fluid phase is not present, the sum of the mole fractions of
 * this fluid phase is smaller than \f$1\f$, i.e.
 * \f[
 * \forall \alpha: S_\alpha = 0 \implies \sum_\kappa x_\alpha^\kappa \leq 1
 * \f]
 *
 * Also, if a fluid phase may be present at a given spatial location
 * its saturation must be positive:
 * \f[ \forall \alpha: \sum_\kappa x_\alpha^\kappa = 1 \implies S_\alpha \geq 0 \f]
 *
 * Since at any given spatial location, a phase is always either
 * present or not present, one of the strict equalities on the
 * right hand side is always true, i.e.
 * \f[ \forall \alpha: S_\alpha \left( \sum_\kappa x_\alpha^\kappa - 1 \right) = 0 \f]
 * always holds.
 *
 * These three equations constitute a non-linear complementarity
 * problem, which can be solved using so-called non-linear
 * complementarity functions \f$\Phi(a, b)\f$ which have the property
 * \f[\Phi(a,b) = 0 \iff a \geq0 \land b \geq0  \land a \cdot b = 0 \f]
 *
 * Several non-linear complementarity functions have been suggested,
 * e.g. the Fischer-Burmeister function
 * \f[ \Phi(a,b) = a + b - \sqrt{a^2 + b^2} \;. \f]
 * This model uses
 * \f[ \Phi(a,b) = \min \{a,  b \}\;, \f]
 * because of its piecewise linearity.
 *
 * These equations are then discretized using a fully-implicit vertex
 * centered finite volume scheme (often known as 'box'-scheme) for
 * spatial discretization and the implicit Euler method as temporal
 * discretization.
 *
 * The model assumes local thermodynamic equilibrium and uses the
 * following primary variables:
 * - The component fugacities \f$f^1, \dots, f^{N}\f$
 * - The pressure of the first phase \f$p_1\f$
 * - The saturations of the first \f$M-1\f$ phases \f$S_1, \dots, S_{M-1}\f$
 * - Temperature \f$T\f$ if the energy equation is enabled
 */

#ifndef DUMUX_MPNC_MODEL_HH
#define DUMUX_MPNC_MODEL_HH

#include <dumux/common/properties.hh>

#include <dumux/material/fluidstates/nonequilibrium.hh>
#include <dumux/material/fluidstates/compositional.hh>
#include <dumux/material/spatialparams/fv.hh>
#include <dumux/material/fluidmatrixinteractions/diffusivitymillingtonquirk.hh>
#include <dumux/material/fluidmatrixinteractions/2p/thermalconductivitysimplefluidlumping.hh>

#include <dumux/porousmediumflow/properties.hh>
#include <dumux/porousmediumflow/compositional/localresidual.hh>
#include <dumux/porousmediumflow/nonisothermal/model.hh>
#include <dumux/porousmediumflow/nonisothermal/indices.hh>
#include <dumux/porousmediumflow/nonisothermal/vtkoutputfields.hh>
#include <dumux/porousmediumflow/nonequilibrium/model.hh>
#include <dumux/porousmediumflow/nonequilibrium/volumevariables.hh>

#include "indices.hh"
#include "volumevariables.hh"
#include "vtkoutputfields.hh"
#include "localresidual.hh"
#include "pressureformulation.hh"

/*!
 * \ingroup MPNCModel
 * \brief Defines the properties required for the MpNc fully implicit model.
 */
namespace Dumux
{

/*!
 * \ingroup MPNCModel
 * \brief Specifies a number properties of the m-phase n-component model.
 *
 * \tparam nPhases the number of phases to be considered
 * \tparam nComp the number of components to be considered
 */
template<int nPhases, int nComp, MpNcPressureFormulation formulation, bool useM, int repCompEqIdx = nComp>
struct MPNCModelTraits
{
    static constexpr int numEq() { return numTransportEq()+numConstraintEq(); }
    static constexpr int numPhases() { return nPhases; }
    static constexpr int numComponents() { return nComp; }
    static constexpr int numTransportEq() { return nComp;}
    static constexpr int numConstraintEq() { return nPhases; }
    static constexpr int replaceCompEqIdx() { return repCompEqIdx; }

    static constexpr bool useMoles() { return useM; }
    static constexpr bool enableAdvection() { return true; }
    static constexpr bool enableMolecularDiffusion() { return true; }
    static constexpr bool enableEnergyBalance() { return false; }
    static constexpr bool enableThermalNonEquilibrium() { return false; }
    static constexpr bool enableChemicalNonEquilibrium() { return false; }

    static constexpr MpNcPressureFormulation pressureFormulation() { return formulation; }

    //! Per default, we use the indices without offset
    using Indices = MPNCIndices< numPhases(), numEq() >;

    template <class FluidSystem, class SolidSystem = void>
    static std::string primaryVariableName(int pvIdx, int state=0)
    {
        if (pvIdx < numComponents())
            return "fugacity^"+ FluidSystem::componentName(pvIdx);
        else if (pvIdx < numEq()-1)
            return "S_"+ FluidSystem::phaseName(pvIdx - numComponents());
        else
        {
            switch (pressureFormulation())
            {
                case MpNcPressureFormulation::mostWettingFirst :
                    return "p_"+ FluidSystem::phaseName(0);
                case MpNcPressureFormulation::leastWettingFirst :
                    return "p_"+ FluidSystem::phaseName(numPhases()-1);
                default: DUNE_THROW(Dune::InvalidStateException, "Invalid formulation ");
            }
        }
    }
};

/*!
 * \ingroup PorousmediumNonEquilibriumModel
 * \brief Specifies a number properties of the m-phase n-component model
 *        in conjunction with non-equilibrium. This is necessary because
 *        the mpnc indices are affected by the non-equilibrium which can
 *        thus not be plugged on top of it that easily.
 *
 * \tparam NonEquilTraits The model traits of the original non-equilibrium model
 */
template<class NonEquilTraits>
struct MPNCNonequilibriumModelTraits : public NonEquilTraits
{
private:
    //! we overwrite the indices as the underlying mpnc indices depend on numTransportEq,
    //! which is again dependent on which form of non-equilibrium is considered
    using MpNcIndices = MPNCIndices< NonEquilTraits::numPhases(),
                                     NonEquilTraits::numTransportEq()+NonEquilTraits::numConstraintEq() >;
public:
    using Indices = NonEquilbriumIndices< MpNcIndices,
                                          NonEquilTraits::numEnergyEqFluid(),
                                          NonEquilTraits::numEnergyEqSolid(),
                                          NonEquilTraits::numEq() >;
};

/*!
 * \ingroup MPNCModel
 * \brief Traits class for the mpnc volume variables.
 *
 * \tparam PV The type used for primary variables
 * \tparam FSY The fluid system type
 * \tparam FST The fluid state type
 * \tparam PT The type used for permeabilities
 * \tparam MT The model traits
 */
template<class PV,
         class FSY,
         class FST,
         class SSY,
         class SST,
         class PT,
         class MT>
struct MPNCVolumeVariablesTraits
{
    using PrimaryVariables = PV;
    using FluidSystem = FSY;
    using FluidState = FST;
    using SolidSystem = SSY;
    using SolidState = SST;
    using PermeabilityType = PT;
    using ModelTraits = MT;
};

namespace Properties
{

//////////////////////////////////////////////////////////////////
// Type tags
//////////////////////////////////////////////////////////////////
//! The type tags for the isothermal & non-isothermal two-phase model
NEW_TYPE_TAG(MPNC, INHERITS_FROM(PorousMediumFlow));
NEW_TYPE_TAG(MPNCNI, INHERITS_FROM(MPNC));
NEW_TYPE_TAG(MPNCNonequil, INHERITS_FROM(MPNC, NonEquilibrium));

/////////////////////////////////////////////////////////////////
// Properties for the isothermal mpnc model
//////////////////////////////////////////////////////////////////

//! Use the MpNc local residual for the MpNc model
SET_TYPE_PROP(MPNC, LocalResidual,  MPNCLocalResidual<TypeTag>);

//! Set the model traits property
SET_PROP(MPNC, ModelTraits)
{
private:
    using FluidSystem = typename GET_PROP_TYPE(TypeTag, FluidSystem);
public:
    using type = MPNCModelTraits<FluidSystem::numPhases,
                                 FluidSystem::numComponents,
                                 GET_PROP_VALUE(TypeTag, PressureFormulation),
                                 GET_PROP_VALUE(TypeTag, UseMoles),
                                 GET_PROP_VALUE(TypeTag, ReplaceCompEqIdx)>;
};

//! This model uses the compositional fluid state
SET_PROP(MPNC, FluidState)
{
private:
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    using FluidSystem = typename GET_PROP_TYPE(TypeTag, FluidSystem);
public:
    using type = CompositionalFluidState<Scalar, FluidSystem>;
};

//! Set the volume variables property
SET_PROP(MPNC, VolumeVariables)
{
private:
    using PV = typename GET_PROP_TYPE(TypeTag, PrimaryVariables);
    using FSY = typename GET_PROP_TYPE(TypeTag, FluidSystem);
    using FST = typename GET_PROP_TYPE(TypeTag, FluidState);
    using SSY = typename GET_PROP_TYPE(TypeTag, SolidSystem);
    using SST = typename GET_PROP_TYPE(TypeTag, SolidState);
    using MT = typename GET_PROP_TYPE(TypeTag, ModelTraits);
    using PT = typename GET_PROP_TYPE(TypeTag, SpatialParams)::PermeabilityType;

    using Traits = MPNCVolumeVariablesTraits<PV, FSY, FST, SSY, SST, PT, MT>;
public:
    using type = MPNCVolumeVariables<Traits>;
};

//! Per default, no component mass balance is replaced
SET_INT_PROP(MPNC, ReplaceCompEqIdx, GET_PROP_TYPE(TypeTag, FluidSystem)::numComponents);
//! Use mole fractions in the balance equations by default
SET_BOOL_PROP(MPNC, UseMoles, true);
//! Use the model after Millington (1961) for the effective diffusivity
SET_TYPE_PROP(MPNC, EffectiveDiffusivityModel, DiffusivityMillingtonQuirk<typename GET_PROP_TYPE(TypeTag, Scalar)>);
//! Set the default pressure formulation to the pressure of the (most) wetting phase
SET_PROP(MPNC, PressureFormulation)
{
public:
    static const MpNcPressureFormulation value = MpNcPressureFormulation::mostWettingFirst;
};

//! Set the vtk output fields specific to this model
SET_TYPE_PROP(MPNC, VtkOutputFields, MPNCVtkOutputFields);

/////////////////////////////////////////////////
// Properties for the non-isothermal mpnc model
/////////////////////////////////////////////////

//! set the non-isothermal model traits
SET_PROP(MPNCNI, ModelTraits)
{
private:
    using FluidSystem = typename GET_PROP_TYPE(TypeTag, FluidSystem);
    using IsothermalTraits = MPNCModelTraits<FluidSystem::numPhases,
                                             FluidSystem::numComponents,
                                             GET_PROP_VALUE(TypeTag, PressureFormulation),
                                             GET_PROP_VALUE(TypeTag, UseMoles),
                                             GET_PROP_VALUE(TypeTag, ReplaceCompEqIdx)>;
public:
    using type = PorousMediumFlowNIModelTraits<IsothermalTraits>;
};

/////////////////////////////////////////////////
// Properties for the non-equilibrium mpnc model
/////////////////////////////////////////////////

SET_TYPE_PROP(MPNCNonequil, EquilibriumLocalResidual, MPNCLocalResidual<TypeTag>);

//! Set the vtk output fields specific to this model
SET_TYPE_PROP(MPNCNonequil, EquilibriumVtkOutputFields, MPNCVtkOutputFields);

//! For non-equilibrium with mpnc we have to overwrite the model traits again,
//! because the mpnc indices depend on the status of the non-equilibrium model traits
SET_PROP(MPNCNonequil, ModelTraits)
{
private:
    using EquiTraits = typename GET_PROP_TYPE(TypeTag, EquilibriumModelTraits);
    static constexpr bool enableTNE = GET_PROP_VALUE(TypeTag, EnableThermalNonEquilibrium);
    static constexpr bool enableCNE = GET_PROP_VALUE(TypeTag, EnableChemicalNonEquilibrium);
    static constexpr int numEF = GET_PROP_VALUE(TypeTag, NumEnergyEqFluid);
    static constexpr int numES = GET_PROP_VALUE(TypeTag, NumEnergyEqSolid);
    static constexpr auto nf = GET_PROP_VALUE(TypeTag, NusseltFormulation);
    static constexpr auto ns = GET_PROP_VALUE(TypeTag, SherwoodFormulation);

    using NonEquilTraits = NonEquilibriumModelTraits<EquiTraits, enableCNE, enableTNE, numEF, numES, nf, ns>;
public:
    using type = MPNCNonequilibriumModelTraits< NonEquilTraits >;
};

//! set equilibrium model traits
SET_PROP(MPNCNonequil, EquilibriumModelTraits)
{
private:
    using FluidSystem = typename GET_PROP_TYPE(TypeTag, FluidSystem);
public:
    using type = MPNCModelTraits<FluidSystem::numPhases,
                                 FluidSystem::numComponents,
                                 GET_PROP_VALUE(TypeTag, PressureFormulation),
                                 GET_PROP_VALUE(TypeTag, UseMoles),
                                 GET_PROP_VALUE(TypeTag, ReplaceCompEqIdx)>;
};

//! in case we do not assume full non-equilibrium one needs a thermal conductivity
SET_PROP(MPNCNonequil, ThermalConductivityModel)
{
private:
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
public:
    using type = ThermalConductivitySimpleFluidLumping<Scalar, GET_PROP_VALUE(TypeTag, NumEnergyEqFluid)>;
};

//! use the mineralization volume variables together with the 2pnc vol vars
SET_PROP(MPNCNonequil, VolumeVariables)
{
private:
    using PV = typename GET_PROP_TYPE(TypeTag, PrimaryVariables);
    using FSY = typename GET_PROP_TYPE(TypeTag, FluidSystem);
    using FST = typename GET_PROP_TYPE(TypeTag, FluidState);
    using SSY = typename GET_PROP_TYPE(TypeTag, SolidSystem);
    using SST = typename GET_PROP_TYPE(TypeTag, SolidState);
    using MT = typename GET_PROP_TYPE(TypeTag, ModelTraits);
    using PT = typename GET_PROP_TYPE(TypeTag, SpatialParams)::PermeabilityType;

    using Traits = MPNCVolumeVariablesTraits<PV, FSY, FST, SSY, SST, PT, MT>;
    using EquilibriumVolVars = MPNCVolumeVariables<Traits>;
public:
    using type = NonEquilibriumVolumeVariables<Traits, EquilibriumVolVars>;
};


} //end namespace Properties
} //end namespace Dumux

#endif