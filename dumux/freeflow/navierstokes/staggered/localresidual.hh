// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
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
 * \ingroup NavierStokesModel
 * \copydoc Dumux::NavierStokesResidualImpl
 */
#ifndef DUMUX_STAGGERED_NAVIERSTOKES_LOCAL_RESIDUAL_HH
#define DUMUX_STAGGERED_NAVIERSTOKES_LOCAL_RESIDUAL_HH

#include <dune/common/hybridutilities.hh>

#include <dumux/assembly/simpleassemblystructs.hh>
#include <dumux/common/properties.hh>
#include <dumux/discretization/method.hh>
#include <dumux/discretization/extrusion.hh>
#include <dumux/assembly/staggeredlocalresidual.hh>
#include <dumux/freeflow/nonisothermal/localresidual.hh>

namespace Dumux {

// forward declaration
template<class TypeTag, DiscretizationMethod discMethod>
class NavierStokesResidualImpl;

/*!
 * \ingroup NavierStokesModel
 * \brief Element-wise calculation of the Navier-Stokes residual for models using the staggered discretization
 */
template<class TypeTag>
class NavierStokesResidualImpl<TypeTag, DiscretizationMethod::staggered>
: public StaggeredLocalResidual<TypeTag>
{
    using ParentType = StaggeredLocalResidual<TypeTag>;
    friend class StaggeredLocalResidual<TypeTag>;

    using GridVariables = GetPropType<TypeTag, Properties::GridVariables>;

    using GridVolumeVariables = typename GridVariables::GridVolumeVariables;
    using ElementVolumeVariables = typename GridVolumeVariables::LocalView;
    using VolumeVariables = typename GridVolumeVariables::VolumeVariables;

    using GridFluxVariablesCache = typename GridVariables::GridFluxVariablesCache;
    using ElementFluxVariablesCache = typename GridFluxVariablesCache::LocalView;

    using GridFaceVariables = typename GridVariables::GridFaceVariables;
    using ElementFaceVariables = typename GridFaceVariables::LocalView;

    using SimpleMassBalanceSummands = GetPropType<TypeTag, Properties::SimpleMassBalanceSummands>;
    using SimpleMomentumBalanceSummands = GetPropType<TypeTag, Properties::SimpleMomentumBalanceSummands>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Implementation = GetPropType<TypeTag, Properties::LocalResidual>;
    using Problem = GetPropType<TypeTag, Properties::Problem>;
    using GridGeometry = GetPropType<TypeTag, Properties::GridGeometry>;
    using FVElementGeometry = typename GridGeometry::LocalView;
    using GridView = typename GridGeometry::GridView;
    using Element = typename GridView::template Codim<0>::Entity;
    using SubControlVolume = typename FVElementGeometry::SubControlVolume;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;
    using Extrusion = Extrusion_t<GridGeometry>;
    using ElementBoundaryTypes = GetPropType<TypeTag, Properties::ElementBoundaryTypes>;
    using CellCenterPrimaryVariables = GetPropType<TypeTag, Properties::CellCenterPrimaryVariables>;
    using FacePrimaryVariables = GetPropType<TypeTag, Properties::FacePrimaryVariables>;
    using FluxVariables = GetPropType<TypeTag, Properties::FluxVariables>;
    using Indices = typename GetPropType<TypeTag, Properties::ModelTraits>::Indices;

    using CellCenterResidual = CellCenterPrimaryVariables;
    using FaceResidual = FacePrimaryVariables;

    using ModelTraits = GetPropType<TypeTag, Properties::ModelTraits>;

public:
    using EnergyLocalResidual = FreeFlowEnergyLocalResidual<GridGeometry, FluxVariables, ModelTraits::enableEnergyBalance(), (ModelTraits::numFluidComponents() > 1)>;

    // account for the offset of the cell center privars within the PrimaryVariables container
    static constexpr auto cellCenterOffset = ModelTraits::numEq() - CellCenterPrimaryVariables::dimension;
    static_assert(cellCenterOffset == ModelTraits::dim(), "cellCenterOffset must equal dim for staggered NavierStokes");

    //! Use the parent type's constructor
    using ParentType::ParentType;

    //! Evaluate fluxes entering or leaving the cell center control volume.
    void computeFluxForCellCenter(const Problem& problem,
                                                        const Element &element,
                                                        const FVElementGeometry& fvGeometry,
                                                        const ElementVolumeVariables& elemVolVars,
                                                        const ElementFaceVariables& elemFaceVars,
                                                        const SubControlVolumeFace &scvf,
                                                        const ElementFluxVariablesCache& elemFluxVarsCache) const
    {
        FluxVariables fluxVars;
        CellCenterPrimaryVariables flux = fluxVars.computeMassFlux(problem, element, fvGeometry, elemVolVars,
                                                                   elemFaceVars, scvf, elemFluxVarsCache[scvf]);

        EnergyLocalResidual::heatFlux(flux, problem, element, fvGeometry, elemVolVars, elemFaceVars, scvf);

        return flux;
    }

    //! Evaluate the source term for the cell center control volume.
    void computeSourceForCellCenter(const Problem& problem,
                                                          const Element &element,
                                                          const FVElementGeometry& fvGeometry,
                                                          const ElementVolumeVariables& elemVolVars,
                                                          const ElementFaceVariables& elemFaceVars,
                                                          const SubControlVolume &scv,
                                    SimpleMassBalanceSummands& simpleMassBalanceSummands) const
    {
        CellCenterPrimaryVariables result(0.0);

        // get the values from the problem
        const auto sourceValues = problem.source(element, fvGeometry, elemVolVars, elemFaceVars, scv);

        // copy the respective cell center related values to the result
        for (int i = 0; i < result.size(); ++i)
            result[i] = sourceValues[i + cellCenterOffset];

        return result;
    }


    //! Evaluate the storage term for the cell center control volume.
    void computeStorageForCellCenter(const Problem& problem,
                                                           const SubControlVolume& scv,
                                                           const VolumeVariables& volVars,
                                     SimpleMassBalanceSummands& simpleMassBalanceSummands) const
    {
        CellCenterPrimaryVariables storage;
        storage[Indices::conti0EqIdx - ModelTraits::dim()] = volVars.density();

        EnergyLocalResidual::fluidPhaseStorage(storage, volVars);

        return storage;
    }

    //! Evaluate the source term for the face control volume.
    void computeSourceForFace(const Problem& problem,
                                              const Element& element,
                                              const FVElementGeometry& fvGeometry,
                                              const SubControlVolumeFace& scvf,
                                              const ElementVolumeVariables& elemVolVars,
                                              const ElementFaceVariables& elemFaceVars,
                                              SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands) const
    {
        FacePrimaryVariables source(0.0);
        const auto& insideVolVars = elemVolVars[scvf.insideScvIdx()];
        source += problem.gravity()[scvf.directionIndex()] * insideVolVars.density();
        source += problem.source(element, fvGeometry, elemVolVars, elemFaceVars, scvf)[Indices::velocity(scvf.directionIndex())];

        return source;
    }

    //! Evaluate the momentum flux for the face control volume.
    void computeFluxForFace(const Problem& problem,
                                            const Element& element,
                                            const SubControlVolumeFace& scvf,
                                            const FVElementGeometry& fvGeometry,
                                            const ElementVolumeVariables& elemVolVars,
                                            const ElementFaceVariables& elemFaceVars,
                                            const ElementFluxVariablesCache& elemFluxVarsCache,
                                            SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands) const
    {
        FluxVariables fluxVars;
        return fluxVars.computeMomentumFlux(problem, element, scvf, fvGeometry, elemVolVars, elemFaceVars, elemFluxVarsCache.gridFluxVarsCache(), simpleMomentumBalanceSummands);
    }

    /*!
     * \brief Evaluate boundary conditions for a cell center dof
     */
    void computeBoundaryFluxForCellCenter(const Problem& problem,
                                                        const Element& element,
                                                        const FVElementGeometry& fvGeometry,
                                                        const SubControlVolumeFace& scvf,
                                                        const ElementVolumeVariables& elemVolVars,
                                                        const ElementFaceVariables& elemFaceVars,
                                                        const ElementBoundaryTypes& elemBcTypes,
                                                        const ElementFluxVariablesCache& elemFluxVarsCache,
                                          SimpleMassBalanceSummands& simpleMassBalanceSummands) const
    {
        CellCenterResidual result(0.0);

        if (scvf.boundary())
        {
            const auto bcTypes = problem.boundaryTypes(element, scvf);

            // no fluxes occur over symmetry boundaries
            if (bcTypes.isSymmetry())
                return;

            // treat Dirichlet and outflow BCs
            result = computeFluxForCellCenter(problem, element, fvGeometry, elemVolVars, elemFaceVars, scvf, elemFluxVarsCache, simpleMassBalanceSummands);

            // treat Neumann BCs, i.e. overwrite certain fluxes by user-specified values
            static constexpr auto numEqCellCenter = CellCenterResidual::dimension;
            if (bcTypes.hasNeumann())
            {
                const auto extrusionFactor = elemVolVars[scvf.insideScvIdx()].extrusionFactor();
                const auto neumannFluxes = problem.neumann(element, fvGeometry, elemVolVars, elemFaceVars, scvf);

                for (int eqIdx = 0; eqIdx < numEqCellCenter; ++eqIdx)
                {
                    if (bcTypes.isNeumann(eqIdx + cellCenterOffset))
                    {
                        result[eqIdx] = 0.0;
                        simpleMassBalanceSummands.RHS[eqIdx] -= neumannFluxes[eqIdx + cellCenterOffset] * extrusionFactor * Extrusion::area(scvf);
                    }
                }
            }

            //non-Dirichlet case
            if(!bcTypes.isDirichlet(Indices::velocity(scvf.directionIndex()))){
                simpleMassBalanceSummands.coefficients[scvf.localFaceIdx()] += boundaryFlux;
            }
            //Dirichlet case
            else{
                simpleMassBalanceSummands.RHS[scvf.localFaceIdx()] -= boundaryFlux;
            }

//             // account for wall functions, if used
//             incorporateWallFunction_(result, problem, element, fvGeometry, scvf, elemVolVars, elemFaceVars);
        }
    }

    /*!
     * \brief Evaluate Dirichlet (fixed value) boundary conditions for a face dof
     */
    void evalDirichletBoundariesForFace(SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands,
                                        const Problem& problem,
                                        const Element& element,
                                        const FVElementGeometry& fvGeometry,
                                        const SubControlVolumeFace& scvf,
                                        const ElementVolumeVariables& elemVolVars,
                                        const ElementFaceVariables& elemFaceVars,
                                        const ElementBoundaryTypes& elemBcTypes,
                                        const ElementFluxVariablesCache& elemFluxVarsCache) const
    {
        if (scvf.boundary())
        {
            // handle the actual boundary conditions:
            const auto bcTypes = problem.boundaryTypes(element, scvf);

            if(bcTypes.isDirichlet(Indices::velocity(scvf.directionIndex())))
            {}
            else if(bcTypes.isSymmetry())
            {
                std::cout << "Symmetry boundary conditions not implemented for SIMPLE." << std::endl;
            }
        }
    }

    /*!
     * \brief Evaluate boundary boundary fluxes for a face dof
     */
    void computeBoundaryFluxForFace(const Problem& problem,
                                            const Element& element,
                                            const FVElementGeometry& fvGeometry,
                                            const SubControlVolumeFace& scvf,
                                            const ElementVolumeVariables& elemVolVars,
                                            const ElementFaceVariables& elemFaceVars,
                                            const ElementBoundaryTypes& elemBcTypes,
                                            const ElementFluxVariablesCache& elemFluxVarsCache,
                                    SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands) const
    {
        FaceResidual result(0.0);

        if (scvf.boundary())
        {
            FluxVariables fluxVars;

            // handle the actual boundary conditions:
            const auto bcTypes = problem.boundaryTypes(element, scvf);
            if (bcTypes.isNeumann(Indices::velocity(scvf.directionIndex())))
            {
                // the source term has already been accounted for, here we
                // add a given Neumann flux for the face on the boundary itself ...
                const auto extrusionFactor = elemVolVars[scvf.insideScvIdx()].extrusionFactor();
                simpleMomentumBalanceSummands.RHS -= problem.neumann(element, fvGeometry, elemVolVars, elemFaceVars, scvf)[Indices::velocity(scvf.directionIndex())]
                                         * extrusionFactor * Extrusion::area(scvf);

                // ... and treat the fluxes of the remaining (frontal and lateral) faces of the staggered control volume
                fluxVars.computeMomentumFlux(problem, element, scvf, fvGeometry, elemVolVars, elemFaceVars, elemFluxVarsCache.gridFluxVarsCache(), simpleMomentumBalanceSummands);
            }
            else if(bcTypes.isDirichlet(Indices::pressureIdx))
            {
                // we are at an "fixed pressure" boundary for which the resdiual of the momentum balance needs to be assembled
                // as if it where inside the domain and not on the boundary (source term has already been acounted for)
                fluxVars.computeMomentumFlux(problem, element, scvf, fvGeometry, elemVolVars, elemFaceVars, elemFluxVarsCache.gridFluxVarsCache(), simpleMomentumBalanceSummands);

                // incorporate the inflow or outflow contribution
                fluxVars.inflowOutflowBoundaryFlux(problem, element, scvf, elemVolVars, elemFaceVars, simpleMomentumBalanceSummands);
            }
        }
    }

private:

    //! do nothing if no turbulence model is used
    template<class ...Args, bool turbulenceModel = ModelTraits::usesTurbulenceModel(), std::enable_if_t<!turbulenceModel, int> = 0>
    void incorporateWallFunction_(Args&&... args) const
    {}

    //! if a turbulence model is used, ask the problem is a wall function shall be employed and get the flux accordingly
    template<bool turbulenceModel = ModelTraits::usesTurbulenceModel(), std::enable_if_t<turbulenceModel, int> = 0>
    void incorporateWallFunction_(CellCenterResidual& boundaryFlux,
                                  const Problem& problem,
                                  const Element& element,
                                  const FVElementGeometry& fvGeometry,
                                  const SubControlVolumeFace& scvf,
                                  const ElementVolumeVariables& elemVolVars,
                                  const ElementFaceVariables& elemFaceVars) const
    {
        static constexpr auto numEqCellCenter = CellCenterResidual::dimension;
        const auto extrusionFactor = elemVolVars[scvf.insideScvIdx()].extrusionFactor();

        // account for wall functions, if used
        for(int eqIdx = 0; eqIdx < numEqCellCenter; ++eqIdx)
        {
            // use a wall function
            if(problem.useWallFunction(element, scvf, eqIdx + cellCenterOffset))
            {
                boundaryFlux[eqIdx] = problem.wallFunction(element, fvGeometry, elemVolVars, elemFaceVars, scvf)[eqIdx]
                                                           * extrusionFactor * Extrusion::area(scvf);
            }
        }
    }

    //! Returns the implementation of the problem (i.e. static polymorphism)
    Implementation &asImp_()
    { return *static_cast<Implementation *>(this); }

    //! \copydoc asImp_()
    const Implementation &asImp_() const
    { return *static_cast<const Implementation *>(this); }
};

} // end namespace Dumux

#endif   // DUMUX_STAGGERED_NAVIERSTOKES_LOCAL_RESIDUAL_HH
