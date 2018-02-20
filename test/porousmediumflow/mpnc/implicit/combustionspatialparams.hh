/*****************************************************************************
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
/*!
 * \file
 * \ingroup MPNCTests
 * \brief Spatialparameters for the combustionproblem1c. Parameters for the actual simulation domain and an outflow region are provided.
 */
#ifndef DUMUX_COMBUSTION_SPATIALPARAMS_HH
#define DUMUX_COMBUSTION_SPATIALPARAMS_HH

#include <dune/common/parametertreeparser.hh>

#include <dumux/material/fluidmatrixinteractions/2p/efftoabslaw.hh>
#include <dumux/material/fluidmatrixinteractions/2p/heatpipelaw.hh>
#include <dumux/material/fluidmatrixinteractions/2p/linearmaterial.hh>
#include <dumux/material/fluidmatrixinteractions/mp/2padapter.hh>
#include <dumux/material/spatialparams/fv.hh>

namespace Dumux
{
/*!
 * \ingroup MPNCTests
 * \brief Spatialparameters for the combustionproblem1c. Parameters for the actual simulation domain and an outflow region are provided.
 */
//forward declaration
template<class TypeTag>
class CombustionSpatialParams;

namespace Properties
{

// The spatial params TypeTag
NEW_TYPE_TAG(CombustionSpatialParams);

// Set the spatial parameters
SET_TYPE_PROP(CombustionSpatialParams, SpatialParams, CombustionSpatialParams<TypeTag>);

// Set the material Law
SET_PROP(CombustionSpatialParams, MaterialLaw)
{
private:
    using FluidSystem = typename GET_PROP_TYPE(TypeTag, FluidSystem);
    enum {wPhaseIdx   = FluidSystem::wPhaseIdx};
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);

//    actually other people call this Leverett
    using EffectiveLaw = HeatPipeLaw<Scalar>;
    using TwoPMaterialLaw = EffToAbsLaw<EffectiveLaw>;
    public:
        using type = TwoPAdapter<wPhaseIdx, TwoPMaterialLaw>;
};


}// end namespace properties

/**
 * \brief Definition of the spatial parameters for the one component combustion problem
 *
 */
template<class TypeTag>
class CombustionSpatialParams : public FVSpatialParams<TypeTag>
{

    using ParentType = FVSpatialParams<TypeTag>;
    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    using GridView = typename GET_PROP_TYPE(TypeTag, GridView);
    using Element = typename GridView::template Codim<0>::Entity;
    using FVElementGeometry = typename GET_PROP_TYPE(TypeTag, FVGridGeometry)::LocalView;
    using SubControlVolume = typename FVElementGeometry::SubControlVolume;
    using ElementSolutionVector = typename GET_PROP_TYPE(TypeTag, ElementSolutionVector);
    using MaterialLaw = typename GET_PROP_TYPE(TypeTag, MaterialLaw);
    using MaterialLawParams = typename MaterialLaw::Params;

    enum {dimWorld = GridView::dimensionworld};
    using GlobalPosition = Dune::FieldVector<Scalar, dimWorld>;

public:
    using PermeabilityType = Scalar;

    CombustionSpatialParams(const Problem &problem)
        : ParentType(problem)
    {
        // this is the parameter value from file part
        porosity_ = getParam<Scalar>("SpatialParams.PorousMedium.porosity");
        intrinsicPermeabilityOutFlow_ = getParam<Scalar>("SpatialParams.Outflow.permeabilityOutFlow");
        porosityOutFlow_                = getParam<Scalar>("SpatialParams.Outflow.porosityOutFlow");
        solidThermalConductivityOutflow_ =getParam<Scalar>("SpatialParams.Outflow.soilThermalConductivityOutFlow");
        solidDensity_   = getParam<Scalar>("SpatialParams.soil.density");
        solidThermalConductivity_ = getParam<Scalar>("SpatialParams.soil.thermalConductivity");
        solidHeatCapacity_   = getParam<Scalar>("SpatialParams.soil.heatCapacity");
        interfacialTension_  = getParam<Scalar>("Constants.interfacialTension");

        Swr_ = getParam<Scalar>("SpatialParams.soil.Swr");
        Snr_ = getParam<Scalar>("SpatialParams.soil.Snr");

        characteristicLength_ =getParam<Scalar>("SpatialParams.PorousMedium.meanPoreSize");

        using std::pow;
        intrinsicPermeability_  =  (pow(characteristicLength_,2.0)  * pow(porosity_, 3.0)) / (150.0 * pow((1.0-porosity_),2.0)); // 1.69e-10 ; //

        factorEnergyTransfer_ = getParam<Scalar>("SpatialParams.PorousMedium.factorEnergyTransfer");
        factorMassTransfer_ =getParam<Scalar>("SpatialParams.PorousMedium.factorMassTransfer");
        lengthPM_ = getParam<Scalar>("Grid.lengthPM");

        // residual saturations
        materialParams_.setSwr(Swr_) ;
        materialParams_.setSnr(Snr_) ;

        using std::sqrt;
        materialParams_.setP0(sqrt(porosity_/intrinsicPermeability_));
        materialParams_.setGamma(interfacialTension_); // interfacial tension of water-air at 100°C
    }

    ~CombustionSpatialParams()
    {}


     PermeabilityType permeability(const Element& element,
                                  const SubControlVolume& scv,
                                  const ElementSolutionVector& elemSol) const
    {
        const auto& globalPos =  scv.dofPosition();
        if ( inOutFlow(globalPos) )
            return intrinsicPermeabilityOutFlow_ ;
        else
            return intrinsicPermeability_ ;
    }

    /*!
     * \brief Define the porosity \f$[-]\f$ of the soil
     *
     * \param element     The finite element
     * \param fvGeometry  The finite volume geometry
     * \param scvIdx      The local index of the sub-control volume where
     *                    the porosity needs to be defined
     */
    Scalar porosity(const Element &element,
                    const SubControlVolume &scv,
                    const ElementSolutionVector &elemSol) const
    {
        const auto& globalPos =  scv.dofPosition();
        if ( inOutFlow(globalPos) )
            return porosityOutFlow_ ;
        else
            return porosity_ ;
    }

    /*!
     * \brief Return a reference to the material parameters of the material law.
     * \param globalPos The position in global coordinates. */
    const MaterialLawParams & materialLawParamsAtPos(const GlobalPosition & globalPos) const
    {
            return materialParams_ ;
    }

    /*!\brief Return the characteristic length for the mass transfer.
     *
     *        The position is determined based on the coordinate of
     *        the vertex belonging to the considered sub controle volume.
     * \param element     The finite element
     * \param fvGeometry  The finite volume geometry
     * \param scvIdx      The local index of the sub control volume */
    const Scalar characteristicLength(const Element & element,
                                      const SubControlVolume &scv,
                                      const ElementSolutionVector &elemSol) const

    {
        const auto& globalPos =  scv.center();
        return characteristicLengthAtPos(globalPos);
    }


    /*!\brief Return the characteristic length for the mass transfer.
     * \param globalPos The position in global coordinates.*/
    const Scalar characteristicLengthAtPos(const  GlobalPosition & globalPos) const
    { return characteristicLength_ ; }

    /*!\brief Return the pre factor the the energy transfer
     *
     *        The position is determined based on the coordinate of
     *        the vertex belonging to the considered sub controle volume.
     * \param element     The finite element
     * \param fvGeometry  The finite volume geometry
     * \param scvIdx      The local index of the sub control volume */
    const Scalar factorEnergyTransfer(const Element &element,
                                      const SubControlVolume &scv,
                                      const ElementSolutionVector &elemSol) const
    {
       const auto& globalPos =  scv.dofPosition();
        return factorEnergyTransferAtPos(globalPos);
    }


    /*!\brief Return the pre factor the the energy transfer
     * \param globalPos The position in global coordinates.*/
    const Scalar factorEnergyTransferAtPos(const  GlobalPosition & globalPos) const
    {
        return factorEnergyTransfer_ ;
    }


    /*!\brief Return the pre factor the the mass transfer
     *
     *        The position is determined based on the coordinate of
     *        the vertex belonging to the considered sub controle volume.
     * \param element     The finite element
     * \param fvGeometry  The finite volume geometry
     * \param scvIdx      The local index of the sub control volume */
    const Scalar factorMassTransfer(const Element &element,
                                    const SubControlVolume &scv,
                                    const ElementSolutionVector &elemSol) const
    {
       const auto& globalPos =  scv.dofPosition();
       return factorMassTransferAtPos(globalPos);
    }

    /*!\brief Return the pre factor the the mass transfer
     * \param globalPos The position in global coordinates.*/
    const Scalar factorMassTransferAtPos(const  GlobalPosition & globalPos) const
    {
        return factorMassTransfer_ ;
    }


    /*!
     * \brief Returns the heat capacity \f$[J / (kg K)]\f$ of the rock matrix.
     *
     * This is only required for non-isothermal models.
     *
     * \param globalPos The global position
     */
    Scalar solidHeatCapacityAtPos(const GlobalPosition& globalPos) const
    {
        return solidHeatCapacity_ ;// specific heat capacity of solid  [J / (kg K)]
    }

    /*!
     * \brief Returns the mass density \f$[kg / m^3]\f$ of the rock matrix.
     *
     * This is only required for non-isothermal models.
     *
     * \param globalPos The global position
     */
    Scalar solidDensityAtPos(const GlobalPosition& globalPos) const
    {
        return solidDensity_ ;// density of solid [kg/m^3]
    }

    /*!
     * \brief Returns the thermal conductivity \f$\mathrm{[W/(m K)]}\f$ of the porous material.
     *
     * This is only required for non-isothermal models.
     *
     * \param globalPos The global position
     */
    Scalar solidThermalConductivity(const Element &element,
                                    const SubControlVolume &scv,
                                    const ElementSolutionVector &elemSol) const
    {
        const auto& globalPos =  scv.dofPosition();
        if ( inOutFlow(globalPos) )
            return solidThermalConductivityOutflow_ ;// conductivity of solid  [W / (m K ) ]
        else
            return solidThermalConductivity_ ;
    }

    /*!
     * \brief Give back whether the tested position (input) is a specific region (right end of porous medium) in the domain
     */
    bool inOutFlow(const GlobalPosition & globalPos) const
    { return globalPos[0] > (lengthPM_ - eps_) ;    }

    /*!
     * \brief Give back how long the porous medium domain is.
     */
    Scalar lengthPM() const
    {
        return lengthPM_ ;
    }

    /*!
     * \brief Give back the itnerfacial tension
     */
    Scalar interfacialTension() const
    {
        return interfacialTension_ ;
    }

private:
    static constexpr Scalar eps_ = 1e-6;

    // Porous Medium Domain
    Scalar intrinsicPermeability_ ;
    Scalar porosity_ ;
    Scalar factorEnergyTransfer_ ;
    Scalar factorMassTransfer_ ;
    Scalar characteristicLength_ ;
    MaterialLawParams   materialParams_ ;

    // Outflow Domain
    Scalar intrinsicPermeabilityOutFlow_ ;
    Scalar porosityOutFlow_ ;

    // solid parameters
    Scalar solidDensity_ ;
    Scalar solidThermalConductivity_ ;
    Scalar solidThermalConductivityOutflow_ ;
    Scalar solidHeatCapacity_ ;
    Scalar interfacialTension_ ;


    // capillary pressures parameters
    Scalar Swr_ ;
    Scalar Snr_ ;

    // grid
    Scalar lengthPM_ ;
};

}

#endif // GUARDIAN
