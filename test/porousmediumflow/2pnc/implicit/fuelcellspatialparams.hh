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
 * \brief Definition of the spatial parameters for the fuel cell
 *        problem which uses the isothermal/non-insothermal 2pnc box model
 */

#ifndef DUMUX_FUELCELL_SPATIAL_PARAMS_HH
#define DUMUX_FUELCELL_SPATIAL_PARAMS_HH

#include <dumux/material/spatialparams/implicit.hh>
#include <dumux/material/fluidmatrixinteractions/2p/linearmaterial.hh>
#include <dumux/material/fluidmatrixinteractions/2p/regularizedbrookscorey.hh>
#include <dumux/material/fluidmatrixinteractions/2p/regularizedvangenuchten.hh>
#include <dumux/material/fluidmatrixinteractions/2p/philtophoblaw.hh>
#include <dumux/porousmediumflow/2pnc/implicit/model.hh>

namespace Dumux
{

//forward declaration
template<class TypeTag>
class FuelCellSpatialParams;

namespace Properties
{
// The spatial parameters TypeTag
NEW_TYPE_TAG(FuelCellSpatialParams);

// Set the spatial parameters
SET_TYPE_PROP(FuelCellSpatialParams, SpatialParams, FuelCellSpatialParams<TypeTag>);

// Set the material Law
SET_PROP(FuelCellSpatialParams, MaterialLaw)
{
 private:
    // define the material law which is parameterized by effective
    // saturations
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef RegularizedVanGenuchten<Scalar> EffMaterialLaw;

 public:
    // define the material law parameterized by absolute saturations
    typedef PhilToPhobLaw<EffMaterialLaw> type;
};
}

/*!
 * \ingroup TwoPTwoCModel
 * \ingroup BoxTestProblems
 * \brief Definition of the spatial parameters for the FuelCell
 *        problem which uses the isothermal 2p2c box model
 */
template<class TypeTag>
class FuelCellSpatialParams : public ImplicitSpatialParams<TypeTag>
{
    typedef ImplicitSpatialParams<TypeTag> ParentType;
    typedef typename GET_PROP_TYPE(TypeTag, Grid) Grid;
    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename Grid::ctype CoordScalar;
    typedef typename GET_PROP_TYPE(TypeTag, FluidSystem) FluidSystem;

    enum {
        dim=GridView::dimension,
        dimWorld=GridView::dimensionworld,

        wPhaseIdx = FluidSystem::wPhaseIdx
    };

    typedef Dune::FieldVector<CoordScalar,dimWorld> GlobalPosition;
    typedef Dune::FieldVector<CoordScalar,dim> DimVector;
    typedef Dune::FieldMatrix<CoordScalar,dim,dim> DimMatrix;

    typedef typename GET_PROP_TYPE(TypeTag, FluxVariables) FluxVariables;
    typedef typename GET_PROP_TYPE(TypeTag, ElementVolumeVariables) ElementVolumeVariables;

    typedef typename GET_PROP_TYPE(TypeTag, FVElementGeometry) FVElementGeometry;
    typedef typename GridView::template Codim<0>::Entity Element;

public:
    typedef typename GET_PROP_TYPE(TypeTag, MaterialLaw) MaterialLaw;
    typedef typename MaterialLaw::Params MaterialLawParams;

    /*!
     * \brief The constructor
     *
     * \param gridView The grid view
     */
    FuelCellSpatialParams(const GridView &gridView)
        : ParentType(gridView), K_(0)
    {
        // intrinsic permeabilities
        K_[0][0] = 5e-11;
        K_[1][1] = 5e-11;

        // porosities
        porosity_ = 0.2;

        // residual saturations
        materialParams_.setSwr(0.12); //here water, see philtophoblaw
        materialParams_.setSnr(0.0);

        //parameters for the vanGenuchten law
        materialParams_.setVgAlpha(6.66e-5); // alpha = 1/pcb
        materialParams_.setVgn(3.652);
    }

    ~FuelCellSpatialParams()
    {}

    /*!
     * \brief Apply the intrinsic permeability tensor to a pressure
     *        potential gradient.
     *
     * \param element The current finite element
     * \param fvGeometry The current finite volume geometry of the element
     * \param scvIdx The index of the sub-control volume
     */
    const DimMatrix intrinsicPermeability(const Element &element,
                                       const FVElementGeometry &fvGeometry,
                                       const int scvIdx) const
    { return K_; }

    /*!
     * \brief Define the porosity \f$[-]\f$ of the spatial parameters
     *
     * \param element The finite element
     * \param fvGeometry The finite volume geometry
     * \param scvIdx The local index of the sub-control volume where
     *                    the porosity needs to be defined
     */
    Scalar porosity(const Element &element,
                    const FVElementGeometry &fvGeometry,
                    const int scvIdx) const
    {
        const GlobalPosition &globalPos = fvGeometry.subContVol[scvIdx].global;

        if (globalPos[1]<eps_)
            return porosity_;
        else
            return 0.2;

    }

    /*!
     * \brief return the parameter object for the Brooks-Corey material law which depends on the position
     *
     * \param element The current finite element
     * \param fvGeometry The current finite volume geometry of the element
     * \param scvIdx The index of the sub-control volume
     */
    const MaterialLawParams& materialLawParams(const Element &element,
                                               const FVElementGeometry &fvGeometry,
                                               const int scvIdx) const
    {
        return materialParams_;
    }

private:
    DimMatrix K_;
    Scalar porosity_;
    static constexpr Scalar eps_ = 1e-6;
    MaterialLawParams materialParams_;
};

}//end namespace

#endif