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
 *
 * \brief Definition of the spatial parameters for the kuevette problem, which
 *        uses the three-phase fully implicit model.
 */
#ifndef DUMUX_INFILTRATION_THREEP_SPATIAL_PARAMS_HH
#define DUMUX_INFILTRATION_THREEP_SPATIAL_PARAMS_HH

#include <dumux/porousmediumflow/3p3c/implicit/indices.hh>
#include <dumux/material/spatialparams/implicit.hh>
#include <dumux/material/fluidmatrixinteractions/3p/regularizedparkervangen3p.hh>
#include <dumux/material/fluidmatrixinteractions/3p/regularizedparkervangen3pparams.hh>
#include <dumux/material/fluidmatrixinteractions/3p/efftoabslaw.hh>
#include <dumux/io/plotmateriallaw3p.hh>
namespace Dumux
{

//forward declaration
template<class TypeTag>
class InfiltrationThreePSpatialParams;

namespace Properties
{
// The spatial parameters TypeTag
NEW_TYPE_TAG(InfiltrationThreePSpatialParams);

// Set the spatial parameters
SET_TYPE_PROP(InfiltrationThreePSpatialParams, SpatialParams, InfiltrationThreePSpatialParams<TypeTag>);

// Set the material Law
SET_PROP(InfiltrationThreePSpatialParams, MaterialLaw)
{
 private:
    // define the material law which is parameterized by effective
    // saturations
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    using EffectiveLaw = RegularizedParkerVanGen3P<Scalar>;
 public:
    // define the material law parameterized by absolute saturations
    using type = EffToAbsLaw<EffectiveLaw>;
};
}

/*!
 * \ingroup ImplicitTestProblems
 * \ingroup ThreePModel
 *
 * \brief Definition of the spatial parameters for the infiltration problem
 */
template<class TypeTag>
class InfiltrationThreePSpatialParams : public ImplicitSpatialParams<TypeTag>
{
    using ParentType = ImplicitSpatialParams<TypeTag>;

    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using GridView = typename GET_PROP_TYPE(TypeTag, GridView);
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    enum {
        dimWorld=GridView::dimensionworld
    };

    using GlobalPosition = Dune::FieldVector<Scalar, GridView::dimension>;

public:
    // export permeability type
    using PermeabilityType = Scalar;

    //get the material law from the property system
    using MaterialLaw = typename GET_PROP_TYPE(TypeTag, MaterialLaw);
    using MaterialLawParams = typename MaterialLaw::Params;

    /*!
     * \brief The constructor
     *
     * \param gridView The grid view
     */
    InfiltrationThreePSpatialParams(const Problem& problem)
        : ParentType(problem)
    {
        // intrinsic permeabilities
        fineK_ = getParam<Scalar>("SpatialParams.permeability");
        coarseK_ = getParam<Scalar>("SpatialParams.permeability");

        // porosities
        porosity_ = getParam<Scalar>("SpatialParams.porosity");

        // residual saturations
        materialParams_.setSwr(0.12);
        materialParams_.setSnr(0.07);
        materialParams_.setSgr(0.03);

        // parameters for the 3phase van Genuchten law
        materialParams_.setVgAlpha(getParam<Scalar>("SpatialParams.vanGenuchtenAlpha"));
        materialParams_.setVgn(getParam<Scalar>("SpatialParams.vanGenuchtenN"));
        materialParams_.setKrRegardsSnr(false);

        // parameters for adsorption
        materialParams_.setKdNAPL(0.);
        materialParams_.setRhoBulk(1500.);

        plotFluidMatrixInteractions_ =  getParam<bool>("Output.PlotFluidMatrixInteractions");
    }

    ~InfiltrationThreePSpatialParams()
    {}

     /*!
     * \brief This is called from the problem and creates a gnuplot output
     *        of e.g the pc-Sw curve
     */
    void plotMaterialLaw()
    {
        PlotMaterialLaw<TypeTag> plotMaterialLaw(plotFluidMatrixInteractions_);

        plotMaterialLaw.plotpc(materialParams_);
        plotMaterialLaw.plotkr(materialParams_);
    }

    /*!
     * \brief Returns the scalar intrinsic permeability \f$[m^2]\f$
     *
     * \param globalPos The global position
     */
    Scalar permeabilityAtPos(const GlobalPosition& globalPos) const
    {
        if (isFineMaterial_(globalPos))
            return fineK_;
        return coarseK_;
    }

    /*!
     * \brief Returns the porosity \f$[-]\f$
     *
     * \param globalPos The global position
     */
    Scalar porosityAtPos(const GlobalPosition& globalPos) const
    {
        return porosity_;
    }

    /*!
     * \brief Returns the parameter object for the Brooks-Corey material law
     *
     * \param globalPos The global position
     */
    const MaterialLawParams& materialLawParamsAtPos(const GlobalPosition& globalPos) const
    {
        return materialParams_;
    }
private:
    bool isFineMaterial_(const GlobalPosition &globalPos) const
    { return
            70. - eps_ <= globalPos[0] && globalPos[0] <= 85. + eps_ &&
            7.0 - eps_ <= globalPos[1] && globalPos[1] <= 7.50 + eps_;
    }

    Scalar fineK_;
    Scalar coarseK_;

    Scalar porosity_;

    MaterialLawParams materialParams_;

    bool plotFluidMatrixInteractions_;

    static constexpr Scalar eps_ = 1e-6;
};

}

#endif
