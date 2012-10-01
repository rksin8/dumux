// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   Copyright (C) 2012 by                                                   *
 *   Institute for Modelling Hydraulic and Environmental Systems             *
 *   University of Stuttgart, Germany                                        *
 *   email: <givenname>.<name>@iws.uni-stuttgart.de                          *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
#ifndef DUMUX_CO2_VOLUME_VARIABLES_HH
#define DUMUX_CO2_VOLUME_VARIABLES_HH

#include <dumux/boxmodels/2p2c/2p2cvolumevariables.hh>

namespace Dumux
{

template <class TypeTag>
class CO2VolumeVariables: public TwoPTwoCVolumeVariables<TypeTag>
{
    typedef TwoPTwoCVolumeVariables<TypeTag> ParentType;
    typedef BoxVolumeVariables<TypeTag> BaseClassType;

    typedef typename GET_PROP_TYPE(TypeTag, VolumeVariables) Implementation;

    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
//    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;
    typedef typename GET_PROP_TYPE(TypeTag, Problem) Problem;
    typedef typename GET_PROP_TYPE(TypeTag, FVElementGeometry) FVElementGeometry;
    typedef typename GET_PROP_TYPE(TypeTag, PrimaryVariables) PrimaryVariables;
    typedef typename GET_PROP_TYPE(TypeTag, FluidSystem) FluidSystem;
    typedef typename GET_PROP_TYPE(TypeTag, MaterialLaw) MaterialLaw;
    typedef typename GET_PROP_TYPE(TypeTag, MaterialLawParams) MaterialLawParams;

    typedef typename GET_PROP_TYPE(TypeTag, Indices) Indices;

    enum {
        numPhases = GET_PROP_VALUE(TypeTag, NumPhases),
        numComponents = GET_PROP_VALUE(TypeTag, NumComponents)
    };

    enum {
        wCompIdx = Indices::wCompIdx,
        nCompIdx = Indices::nCompIdx,
        wPhaseIdx = Indices::wPhaseIdx,
        nPhaseIdx = Indices::nPhaseIdx
    };

    // present phases
    enum {
        wPhaseOnly = Indices::lPhaseOnly,
        nPhaseOnly = Indices::gPhaseOnly,
        bothPhases = Indices::bothPhases
    };

    // formulations
    enum {
        formulation = GET_PROP_VALUE(TypeTag, Formulation),
        pwSn = TwoPTwoCFormulation::pwSn,
        pnSw = TwoPTwoCFormulation::pnSw
    };

    // primary variable indices
    enum {
        switchIdx = Indices::switchIdx,
        pressureIdx = Indices::pressureIdx
    };

    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;
    typedef typename GridView::template Codim<0>::Entity Element;
    enum { dim = GridView::dimension};

    static const Scalar R; // universial nonwetting constant

public:
    //! The type of the object returned by the fluidState() method
    typedef Dumux::CompositionalFluidState<Scalar, FluidSystem> FluidState;


    /*!
     * \brief Update all quantities for a given control volume.
     *
     * \param priVars The primary variables
     * \param problem The problem
     * \param element The element
     * \param fvGeometry The finite-volume geometry in the box scheme
     * \param scvIdx The local index of the SCV (sub-control volume)
     * \param isOldSol Evaluate function with solution of current or previous time step
     */
    void update(const PrimaryVariables &priVars,
                const Problem &problem,
                const Element &element,
                const FVElementGeometry &fvGeometry,
                const int scvIdx,
                const bool isOldSol)
    {
    	// Update BoxVolVars but not 2p2cvolvars
        // ToDo: Is BaseClassType the right name?
        BaseClassType::update(priVars,
                problem,
                element,
                fvGeometry,
                scvIdx,
                isOldSol);

        unsigned numDofs = problem.model().numDofs();
        unsigned numVertices = problem.gridView().size(dim);

        int globalIdx;
        if (numDofs != numVertices) // element data
            globalIdx = problem.model().dofMapper().map(element);
        else
            globalIdx = problem.model().dofMapper().map(element, scvIdx, dim);

        int phasePresence = problem.model().phasePresence(globalIdx, isOldSol);

         Scalar temp = Implementation::temperature_(priVars, problem, element, fvGeometry, scvIdx);
         ParentType::fluidState_.setTemperature(temp);

         /////////////
         // set the saturations
         /////////////
         Scalar Sn;
         if (phasePresence == nPhaseOnly)
             Sn = 1.0;
         else if (phasePresence == wPhaseOnly) {
             Sn = 0.0;
         }
         else if (phasePresence == bothPhases) {
             if (formulation == pwSn)
                 Sn = priVars[switchIdx];
             else if (formulation == pnSw)
                 Sn = 1.0 - priVars[switchIdx];
             else DUNE_THROW(Dune::InvalidStateException, "Formulation: " << formulation << " is invalid.");
         }
         else DUNE_THROW(Dune::InvalidStateException, "phasePresence: " << phasePresence << " is invalid.");
         ParentType::fluidState_.setSaturation(wPhaseIdx, 1 - Sn);
         ParentType::fluidState_.setSaturation(nPhaseIdx, Sn);

         // capillary pressure parameters
          const MaterialLawParams &materialParams =
              problem.spatialParams().materialLawParams(element, fvGeometry, scvIdx);


          Scalar pC = MaterialLaw::pC(materialParams, 1 - Sn);

          if (formulation == pwSn) {
              ParentType::fluidState_.setPressure(wPhaseIdx, priVars[pressureIdx]);
              ParentType::fluidState_.setPressure(nPhaseIdx, priVars[pressureIdx] + pC);
          }
          else if (formulation == pnSw) {
              ParentType::fluidState_.setPressure(nPhaseIdx, priVars[pressureIdx]);
              ParentType::fluidState_.setPressure(wPhaseIdx, priVars[pressureIdx] - pC);
          }
          else DUNE_THROW(Dune::InvalidStateException, "Formulation: " << formulation << " is invalid.");

          /////////////
          // calculate the phase compositions
          /////////////
          typename FluidSystem::ParameterCache paramCache;


          // calculate phase composition
          if (phasePresence == bothPhases) {

              //Get the equilibrium mole fractions from the FluidSystem and set them in the fluidState
              //xCO2 = equilibrium mole fraction of CO2 in the liquid phase
              //yH2O = equilibrium mole fraction of H2O in the gas phase

              Scalar xwCO2 = FluidSystem::equilibriumMoleFraction(ParentType::fluidState_, paramCache, wPhaseIdx);
              Scalar xgH2O = FluidSystem::equilibriumMoleFraction(ParentType::fluidState_, paramCache, nPhaseIdx);
              Scalar xwH2O = 1 - xwCO2;
              Scalar xgCO2 = 1 - xgH2O;

              ParentType::fluidState_.setMoleFraction(wPhaseIdx, wCompIdx, xwH2O);
              ParentType::fluidState_.setMoleFraction(wPhaseIdx, nCompIdx, xwCO2);
              ParentType::fluidState_.setMoleFraction(nPhaseIdx, wCompIdx, xgH2O);
              ParentType::fluidState_.setMoleFraction(nPhaseIdx, nCompIdx, xgCO2);


              //Get the phase densities from the FluidSystem and set them in the fluidState

              Scalar rhoW = FluidSystem::density(ParentType::fluidState_, paramCache, wPhaseIdx);
              Scalar rhoN = FluidSystem::density(ParentType::fluidState_, paramCache, nPhaseIdx);

              ParentType::fluidState_.setDensity(wPhaseIdx, rhoW);
              ParentType::fluidState_.setDensity(nPhaseIdx, rhoN);


              //Get the phase viscosities from the FluidSystem and set them in the fluidState

              Scalar muW = FluidSystem::viscosity(ParentType::fluidState_, paramCache, wPhaseIdx);
              Scalar muN = FluidSystem::viscosity(ParentType::fluidState_, paramCache, nPhaseIdx);

              ParentType::fluidState_.setViscosity(wPhaseIdx, muW);
              ParentType::fluidState_.setViscosity(nPhaseIdx, muN);

          }
          else if (phasePresence == nPhaseOnly) {
              // only the nonwetting phase is present, i.e. nonwetting phase
              // composition is stored explicitly.

              // extract _mass_ fractions in the nonwetting phase
              Scalar massFractionN[numComponents];
              massFractionN[wCompIdx] = priVars[switchIdx];
              massFractionN[nCompIdx] = 1 - massFractionN[wCompIdx];

              // calculate average molar mass of the nonwetting phase
              Scalar M1 = FluidSystem::molarMass(wCompIdx);
              Scalar M2 = FluidSystem::molarMass(nCompIdx);
              Scalar X2 = massFractionN[nCompIdx];
              Scalar avgMolarMass = M1*M2/(M2 + X2*(M1 - M2));

              // convert mass to mole fractions and set the fluid state
              ParentType::fluidState_.setMoleFraction(nPhaseIdx, wCompIdx, massFractionN[wCompIdx]*avgMolarMass/M1);
              ParentType::fluidState_.setMoleFraction(nPhaseIdx, nCompIdx, massFractionN[nCompIdx]*avgMolarMass/M2);

              // TODO give values for non-existing wetting phase
              Scalar xwCO2 = FluidSystem::equilibriumMoleFraction(ParentType::fluidState_, paramCache, wPhaseIdx);
              Scalar xwH2O = 1 - xwCO2;
//              Scalar xwCO2 = FluidSystem::equilibriumMoleFraction(ParentType::fluidState_, paramCache, wPhaseIdx, nPhaseOnly);
//              Scalar xwH2O = 1 - xwCO2;
              ParentType::fluidState_.setMoleFraction(wPhaseIdx, nCompIdx, xwCO2);
              ParentType::fluidState_.setMoleFraction(wPhaseIdx, wCompIdx, xwH2O);


              //Get the phase densities from the FluidSystem and set them in the fluidState

              Scalar rhoW = FluidSystem::density(ParentType::fluidState_, paramCache, wPhaseIdx);
              Scalar rhoN = FluidSystem::density(ParentType::fluidState_, paramCache, nPhaseIdx);

              ParentType::fluidState_.setDensity(wPhaseIdx, rhoW);
              ParentType::fluidState_.setDensity(nPhaseIdx, rhoN);

              //Get the phase viscosities from the FluidSystem and set them in the fluidState

              Scalar muW = FluidSystem::viscosity(ParentType::fluidState_, paramCache, wPhaseIdx);
              Scalar muN = FluidSystem::viscosity(ParentType::fluidState_, paramCache, nPhaseIdx);

              ParentType::fluidState_.setViscosity(wPhaseIdx, muW);
              ParentType::fluidState_.setViscosity(nPhaseIdx, muN);
          }
          else if (phasePresence == wPhaseOnly) {
               // only the wetting phase is present, i.e. wetting phase
               // composition is stored explicitly.

               // extract _mass_ fractions in the nonwetting phase
               Scalar massFractionW[numComponents];
               massFractionW[nCompIdx] = priVars[switchIdx];
               massFractionW[wCompIdx] = 1 - massFractionW[nCompIdx];

               // calculate average molar mass of the nonwetting phase
               Scalar M1 = FluidSystem::molarMass(wCompIdx);
               Scalar M2 = FluidSystem::molarMass(nCompIdx);
               Scalar X2 = massFractionW[nCompIdx];
               Scalar avgMolarMass = M1*M2/(M2 + X2*(M1 - M2));

               // convert mass to mole fractions and set the fluid state
               ParentType::fluidState_.setMoleFraction(wPhaseIdx, wCompIdx, massFractionW[wCompIdx]*avgMolarMass/M1);
               ParentType::fluidState_.setMoleFraction(wPhaseIdx, nCompIdx, massFractionW[nCompIdx]*avgMolarMass/M2);

               //  TODO give values for non-existing nonwetting phase
               Scalar xnH2O = FluidSystem::equilibriumMoleFraction(ParentType::fluidState_, paramCache, nPhaseIdx);
               Scalar xnCO2 = 1 - xnH2O; //FluidSystem::equilibriumMoleFraction(ParentType::fluidState_, paramCache, nPhaseIdx);
//               Scalar xnH2O = FluidSystem::equilibriumMoleFraction(ParentType::fluidState_, paramCache, nPhaseIdx, wPhaseOnly);
//               Scalar xnCO2 = 1 - xnH2O;
               ParentType::fluidState_.setMoleFraction(nPhaseIdx, nCompIdx, xnCO2);
               ParentType::fluidState_.setMoleFraction(nPhaseIdx, wCompIdx, xnH2O);


               Scalar rhoW = FluidSystem::density(ParentType::fluidState_, paramCache, wPhaseIdx);
               Scalar rhoN = FluidSystem::density(ParentType::fluidState_, paramCache, nPhaseIdx);

               ParentType::fluidState_.setDensity(wPhaseIdx, rhoW);
               ParentType::fluidState_.setDensity(nPhaseIdx, rhoN);

               //Get the phase viscosities from the FluidSystem and set them in the fluidState

               Scalar muW = FluidSystem::viscosity(ParentType::fluidState_, paramCache, wPhaseIdx);
               Scalar muN = FluidSystem::viscosity(ParentType::fluidState_, paramCache, nPhaseIdx);

               ParentType::fluidState_.setViscosity(wPhaseIdx, muW);
               ParentType::fluidState_.setViscosity(nPhaseIdx, muN);
           }

          for (int phaseIdx = 0; phaseIdx < numPhases; ++phaseIdx) {
              // compute and set the enthalpy
              Scalar h = Implementation::enthalpy_(ParentType::fluidState_, paramCache, phaseIdx);
              ParentType::fluidState_.setEnthalpy(phaseIdx, h);
          }

          paramCache.updateAll(ParentType::fluidState_);

          for (int phaseIdx = 0; phaseIdx < numPhases; ++phaseIdx) {
              // relative permeabilities
              Scalar kr;
              if (phaseIdx == wPhaseIdx)
                  kr = MaterialLaw::krw(materialParams, this->saturation(wPhaseIdx));
              else // ATTENTION: krn requires the liquid saturation
                  // as parameter!
                  kr = MaterialLaw::krn(materialParams, this->saturation(wPhaseIdx));
              ParentType::relativePermeability_[phaseIdx] = kr;
              Valgrind::CheckDefined(ParentType::relativePermeability_[phaseIdx]);

              // binary diffusion coefficents
              ParentType::diffCoeff_[phaseIdx] =
                  FluidSystem::binaryDiffusionCoefficient(ParentType::fluidState_,
                                                          paramCache,
                                                          phaseIdx,
                                                          wCompIdx,
                                                          nCompIdx);
              Valgrind::CheckDefined(ParentType::diffCoeff_[phaseIdx]);
          }

          // porosity
          ParentType::porosity_ = problem.spatialParams().porosity(element,
                                                           fvGeometry,
                                                           scvIdx);
          Valgrind::CheckDefined(ParentType::porosity_);
//          if(phasePresence == bothPhases)
//          {
//              std::cout<<"globalIdx = "<<globalIdx<<std::endl;
//              std::cout<<"scvIdx = "<<globalIdx<<std::endl;
//              std::cout<<"Sn = "<<ParentType::fluidState_.saturation(nPhaseIdx)<<std::endl;
//              std::cout<<"Sw = "<<ParentType::fluidState_.saturation(wPhaseIdx)<<std::endl;
//              std::cout<<"mobilityN = "<<ParentType::mobility(nPhaseIdx)<<std::endl;
//              std::cout<<"xgH2O = "<<ParentType::fluidState_.moleFraction(nPhaseIdx, wCompIdx)<<std::endl;
//              std::cout<<"xgCO2 = "<<ParentType::fluidState_.moleFraction(nPhaseIdx, nCompIdx)<<std::endl;
//              std::cout<<"xwH2O = "<<ParentType::fluidState_.moleFraction(wPhaseIdx, wCompIdx)<<std::endl;
//              std::cout<<"xwCO2 = "<<ParentType::fluidState_.moleFraction(wPhaseIdx, nCompIdx)<<std::endl;
//              std::cout<<"XgH2O = "<<ParentType::fluidState_.massFraction(nPhaseIdx, wCompIdx)<<std::endl;
//              std::cout<<"XgCO2 = "<<ParentType::fluidState_.massFraction(nPhaseIdx, nCompIdx)<<std::endl;
//              std::cout<<"XwH2O = "<<ParentType::fluidState_.massFraction(wPhaseIdx, wCompIdx)<<std::endl;
//              std::cout<<"XwCO2 = "<<ParentType::fluidState_.massFraction(wPhaseIdx, nCompIdx)<<std::endl;
//          }

          // energy related quantities not contained in the fluid state
          asImp_().updateEnergy_(priVars, problem, element, fvGeometry, scvIdx, isOldSol);
    }




protected:

    static Scalar temperature_(const PrimaryVariables &priVars,
                               const Problem& problem,
                               const Element &element,
                               const FVElementGeometry &fvGeometry,
                               int scvIdx)
    {
        return problem.boxTemperature(element, fvGeometry, scvIdx);
    }

    template<class ParameterCache>
    static Scalar enthalpy_(const FluidState& fluidState,
                            const ParameterCache& paramCache,
                            const int phaseIdx)
    {
        return 0;
    }

    void updateEnergy_(const PrimaryVariables &sol,
                       const Problem &problem,
                       const Element &element,
                       const FVElementGeometry &fvGeometry,
                       const int vertIdx,
                       bool isOldSol)
    { }



private:



    Implementation &asImp_()
    { return *static_cast<Implementation*>(this); }

    const Implementation &asImp_() const
    { return *static_cast<const Implementation*>(this); }
};

} // end namepace

#endif
