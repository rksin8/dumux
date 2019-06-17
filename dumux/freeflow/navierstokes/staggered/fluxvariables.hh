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
 * \copydoc Dumux::NavierStokesFluxVariablesImpl
 */
#ifndef DUMUX_NAVIERSTOKES_STAGGERED_FLUXVARIABLES_HH
#define DUMUX_NAVIERSTOKES_STAGGERED_FLUXVARIABLES_HH

#include <array>
#include <dune/common/std/optional.hh>

#include <dumux/common/math.hh>
#include <dumux/common/exceptions.hh>
#include <dumux/common/parameters.hh>
#include <dumux/common/properties.hh>

#include <dumux/flux/fluxvariablesbase.hh>
#include <dumux/discretization/method.hh>


#include "staggeredupwindfluxvariables.hh"

namespace Dumux {

// forward declaration
template<class TypeTag, DiscretizationMethod discMethod>
class NavierStokesFluxVariablesImpl;

/*!
 * \ingroup NavierStokesModel
 * \brief The flux variables class for the Navier-Stokes model using the staggered grid discretization.
 */
template<class TypeTag>
class NavierStokesFluxVariablesImpl<TypeTag, DiscretizationMethod::staggered>
: public FluxVariablesBase<GetPropType<TypeTag, Properties::Problem>,
                           typename GetPropType<TypeTag, Properties::FVGridGeometry>::LocalView,
                           typename GetPropType<TypeTag, Properties::GridVolumeVariables>::LocalView,
                           typename GetPropType<TypeTag, Properties::GridFluxVariablesCache>::LocalView>
{
    using GridVariables = GetPropType<TypeTag, Properties::GridVariables>;

    using GridVolumeVariables = typename GridVariables::GridVolumeVariables;
    using ElementVolumeVariables = typename GridVolumeVariables::LocalView;
    using VolumeVariables = typename GridVolumeVariables::VolumeVariables;

    using GridFluxVariablesCache = typename GridVariables::GridFluxVariablesCache;
    using FluxVariablesCache = typename GridFluxVariablesCache::FluxVariablesCache;

    using GridFaceVariables = typename GridVariables::GridFaceVariables;
    using ElementFaceVariables = typename GridFaceVariables::LocalView;
    using FaceVariables = typename GridFaceVariables::FaceVariables;

    using Problem = GetPropType<TypeTag, Properties::Problem>;
    using FVGridGeometry = GetPropType<TypeTag, Properties::FVGridGeometry>;
    using FVElementGeometry = typename FVGridGeometry::LocalView;
    using GridView = typename FVGridGeometry::GridView;
    using Element = typename GridView::template Codim<0>::Entity;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using ModelTraits = GetPropType<TypeTag, Properties::ModelTraits>;
    using Indices = typename ModelTraits::Indices;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;
    using CellCenterPrimaryVariables = GetPropType<TypeTag, Properties::CellCenterPrimaryVariables>;
    using FacePrimaryVariables = GetPropType<TypeTag, Properties::FacePrimaryVariables>;
    using BoundaryTypes = GetPropType<TypeTag, Properties::BoundaryTypes>;

    static constexpr bool normalizePressure = getPropValue<TypeTag, Properties::NormalizePressure>();

    using GlobalPosition = typename Element::Geometry::GlobalCoordinate;

    static constexpr auto cellCenterIdx = FVGridGeometry::cellCenterIdx();
    static constexpr auto faceIdx = FVGridGeometry::faceIdx();

    static constexpr int upwindSchemeOrder = getPropValue<TypeTag, Properties::UpwindSchemeOrder>();
    static constexpr bool useHigherOrder = upwindSchemeOrder > 1;

public:

    using HeatConductionType = GetPropType<TypeTag, Properties::HeatConductionType>;

    /*!
     * \brief Returns the advective flux over a sub control volume face.
     * \param problem The object specifying the problem which ought to be simulated
     * \param elemVolVars All volume variables for the element
     * \param elemFaceVars The face variables
     * \param scvf The sub control volume face
     * \param upwindTerm The uwind term (i.e. the advectively transported quantity)
     */
    template<class UpwindTerm>
    static Scalar advectiveFluxForCellCenter(const Problem& problem,
                                             const ElementVolumeVariables& elemVolVars,
                                             const ElementFaceVariables& elemFaceVars,
                                             const SubControlVolumeFace &scvf,
                                             UpwindTerm upwindTerm)
    {
        const Scalar velocity = elemFaceVars[scvf].velocitySelf();
        const bool insideIsUpstream = scvf.directionSign() == sign(velocity);
        static const Scalar upwindWeight = getParamFromGroup<Scalar>(problem.paramGroup(), "Flux.UpwindWeight");

        const auto& insideVolVars = elemVolVars[scvf.insideScvIdx()];
        const auto& outsideVolVars = elemVolVars[scvf.outsideScvIdx()];

        const auto& upstreamVolVars = insideIsUpstream ? insideVolVars : outsideVolVars;
        const auto& downstreamVolVars = insideIsUpstream ? outsideVolVars : insideVolVars;

        const Scalar flux = (upwindWeight * upwindTerm(upstreamVolVars) +
                            (1.0 - upwindWeight) * upwindTerm(downstreamVolVars))
                            * velocity * scvf.area() * scvf.directionSign();

        return flux * extrusionFactor_(elemVolVars, scvf);
    }

    /*!
     * \brief Computes the flux for the cell center residual (mass balance).
     *
     * \verbatim
     *                    scvf
     *              ----------------
     *              |              # current scvf    # scvf over which fluxes are calculated
     *              |              #
     *              |      x       #~~~~> vel.Self   x dof position
     *              |              #
     *        scvf  |              #                 -- element
     *              ----------------
     *                   scvf
     * \endverbatim
     */
    CellCenterPrimaryVariables computeMassFlux(const Problem& problem,
                                               const Element& element,
                                               const FVElementGeometry& fvGeometry,
                                               const ElementVolumeVariables& elemVolVars,
                                               const ElementFaceVariables& elemFaceVars,
                                               const SubControlVolumeFace& scvf,
                                               const FluxVariablesCache& fluxVarsCache)
    {
        // The advectively transported quantity (i.e density for a single-phase system).
        auto upwindTerm = [](const auto& volVars) { return volVars.density(); };

        // Call the generic flux function.
        CellCenterPrimaryVariables result(0.0);
        result[Indices::conti0EqIdx - ModelTraits::dim()] = advectiveFluxForCellCenter(problem, elemVolVars, elemFaceVars, scvf, upwindTerm);

        return result;
    }

    /*!
     * \brief Returns the momentum flux over all staggered faces.
     */
    FacePrimaryVariables computeMomentumFlux(const Problem& problem,
                                             const Element& element,
                                             const SubControlVolumeFace& scvf,
                                             const FVElementGeometry& fvGeometry,
                                             const ElementVolumeVariables& elemVolVars,
                                             const ElementFaceVariables& elemFaceVars,
                                             const GridFluxVariablesCache& gridFluxVarsCache)
    {
        return computeFrontalMomentumFlux(problem, element, scvf, fvGeometry, elemVolVars, elemFaceVars, gridFluxVarsCache) +
               computeLateralMomentumFlux(problem, element, scvf, fvGeometry, elemVolVars, elemFaceVars, gridFluxVarsCache);
    }

    /*!
     * \brief Returns the frontal part of the momentum flux.
     *        This treats the flux over the staggered face at the center of an element,
     *        parallel to the current scvf where the velocity dof of interest lives.
     *
     * \verbatim
     *                    scvf
     *              ---------=======                 == and # staggered half-control-volume
     *              |       #      | current scvf
     *              |       #      |                 # staggered face over which fluxes are calculated
     *   vel.Opp <~~|       #~~>   x~~~~> vel.Self
     *              |       #      |                 x dof position
     *        scvf  |       #      |
     *              --------========                 -- element
     *                   scvf
     * \endverbatim
     */
    FacePrimaryVariables computeFrontalMomentumFlux(const Problem& problem,
                                                    const Element& element,
                                                    const SubControlVolumeFace& scvf,
                                                    const FVElementGeometry& fvGeometry,
                                                    const ElementVolumeVariables& elemVolVars,
                                                    const ElementFaceVariables& elemFaceVars,
                                                    const GridFluxVariablesCache& gridFluxVarsCache)
    {
        FacePrimaryVariables frontalFlux(0.0);

        // The velocities of the dof at interest and the one of the opposite scvf.
        const Scalar velocitySelf = elemFaceVars[scvf].velocitySelf();
        const Scalar velocityOpposite = elemFaceVars[scvf].velocityOpposite();

        // Advective flux.
        if (problem.enableInertiaTerms())
        {
            // Get the average velocity at the center of the element (i.e. the location of the staggered face).
            const Scalar transportingVelocity = (velocitySelf + velocityOpposite) * 0.5;

            frontalFlux += StaggeredUpwindFluxVariables<TypeTag, upwindSchemeOrder>::computeUpwindedFrontalMomentum(scvf, elemFaceVars, elemVolVars, gridFluxVarsCache, transportingVelocity)
                           * transportingVelocity * -1.0 * scvf.directionSign();
        }

        // The volume variables within the current element. We only require those (and none of neighboring elements)
        // because the fluxes are calculated over the staggered face at the center of the element.
        const auto& insideVolVars = elemVolVars[scvf.insideScvIdx()];

        // Diffusive flux.
        // The velocity gradient already accounts for the orientation
        // of the staggered face's outer normal vector.
        const Scalar velocityGrad_ii = (velocityOpposite - velocitySelf) / scvf.selfToOppositeDistance();

        static const bool enableUnsymmetrizedVelocityGradient
            = getParamFromGroup<bool>(problem.paramGroup(), "FreeFlow.EnableUnsymmetrizedVelocityGradient", false);
        const Scalar factor = enableUnsymmetrizedVelocityGradient ? 1.0 : 2.0;
        frontalFlux -= factor * insideVolVars.effectiveViscosity() * velocityGrad_ii;

        // The pressure term.
        // If specified, the pressure can be normalized using the initial value on the scfv of interest.
        // The scvf is used to normalize by the same value from the left and right side.
        // Can potentially help to improve the condition number of the system matrix.
        const Scalar pressure = normalizePressure ?
                                insideVolVars.pressure() - problem.initial(scvf)[Indices::pressureIdx]
                              : insideVolVars.pressure();

        // Account for the orientation of the staggered face's normal outer normal vector
        // (pointing in opposite direction of the scvf's one).
        frontalFlux += pressure * -1.0 * scvf.directionSign();

        // Account for the staggered face's area. For rectangular elements, this equals the area of the scvf
        // our velocity dof of interest lives on.
        return frontalFlux * scvf.area() * insideVolVars.extrusionFactor();
   }

    /*!
     * \brief Returns the momentum flux over the staggered faces
     *        perpendicular to the scvf where the velocity dof of interest
     *         lives (coinciding with the element's scvfs).
     *
     * \verbatim
     *                scvf
     *              ---------#######                 || and # staggered half-control-volume
     *              |      ||      | current scvf
     *              |      ||      |                 # normal staggered sub faces over which fluxes are calculated
     *              |      ||      x~~~~> vel.Self
     *              |      ||      |                 x dof position
     *        scvf  |      ||      |
     *              --------########                -- element
     *                 scvf
     * \endverbatim
     */
    FacePrimaryVariables computeLateralMomentumFlux(const Problem& problem,
                                                    const Element& element,
                                                    const SubControlVolumeFace& scvf,
                                                    const FVElementGeometry& fvGeometry,
                                                    const ElementVolumeVariables& elemVolVars,
                                                    const ElementFaceVariables& elemFaceVars,
                                                    const GridFluxVariablesCache& gridFluxVarsCache)
    {
        FacePrimaryVariables lateralFlux(0.0);
        const auto& faceVars = elemFaceVars[scvf];
        const int numSubFaces = scvf.pairData().size();

        // If the current scvf is on a boundary, check if there is a Neumann BC for the stress in tangential direction.
        // Create a boundaryTypes object (will be empty if not at a boundary).
        Dune::Std::optional<BoundaryTypes> currentScvfBoundaryTypes;
        if (scvf.boundary())
            currentScvfBoundaryTypes.emplace(problem.boundaryTypes(element, scvf));

        // Account for all sub faces.
        for (int localSubFaceIdx = 0; localSubFaceIdx < numSubFaces; ++localSubFaceIdx)
        {
            const auto eIdx = scvf.insideScvIdx();
            // Get the face normal to the face the dof lives on. The staggered sub face conincides with half of this lateral face.
            const auto& lateralScvf = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);

            // If the current scvf is on a bounary and if there is a Neumann BC for the stress in tangential direction,
            // assign this value for the lateral momentum flux. No further calculations are required. We assume that all lateral faces
            // have the same type of BC (Neumann), but we sample the value at their actual positions.
            if (currentScvfBoundaryTypes && currentScvfBoundaryTypes->isNeumann(Indices::velocity(lateralScvf.directionIndex())))
            {
                // Get the location of the lateral staggered face's center.
                const auto& lateralStaggeredFaceCenter = lateralStaggeredFaceCenter_(scvf, localSubFaceIdx);
                lateralFlux += problem.neumann(element, fvGeometry, elemVolVars, elemFaceVars,
                                               scvf.makeBoundaryFace(lateralStaggeredFaceCenter))[Indices::velocity(lateralScvf.directionIndex())]
                                              * elemVolVars[scvf.insideScvIdx()].extrusionFactor() * lateralScvf.area() * 0.5
                                              * lateralScvf.directionSign();
                continue;
            }

            // Create a boundaryTypes object (will be empty if not at a boundary).
            Dune::Std::optional<BoundaryTypes> lateralFaceBoundaryTypes;

            // Check if there is face/element parallel to our face of interest where the dof lives on. If there is no parallel neighbor,
            // we are on a boundary where we have to check for boundary conditions.
            if (lateralScvf.boundary())
            {
                // Retrieve the boundary types that correspond to the center of the lateral scvf. As a convention, we always query
                // the type of BCs at the center of the element's "actual" lateral scvf (not the face of the staggered control volume).
                // The value of the BC will be evaluated at the center of the staggered face.
                //     --------T######V                 || frontal face of staggered half-control-volume
                //     |      ||      | current scvf    #  lateral staggered face of interest (may lie on a boundary)
                //     |      ||      |                 x  dof position
                //     |      ||      x~~~~> vel.Self   -- element boundaries
                //     |      ||      |                 T  position at which the type of boundary conditions will be evaluated
                //     |      ||      |                    (center of lateral scvf)
                //     ----------------                 V  position at which the value of the boundary conditions will be evaluated
                //                                         (center of the staggered lateral face)
                lateralFaceBoundaryTypes.emplace(problem.boundaryTypes(element, lateralScvf));

                // Check if we have a symmetry boundary condition. If yes, the tangential part of the momentum flux can be neglected
                // and we may skip any further calculations for the given sub face.
                if (lateralFaceBoundaryTypes->isSymmetry())
                    continue;

                // Handle Neumann boundary conditions. No further calculations are then required for the given sub face.
                if (lateralFaceBoundaryTypes->isNeumann(Indices::velocity(scvf.directionIndex())))
                {
                    // Construct a temporary scvf which corresponds to the staggered sub face, featuring the location
                    // the staggered faces's center.
                    const auto& lateralStaggeredFaceCenter = lateralStaggeredFaceCenter_(scvf, localSubFaceIdx);
                    const auto lateralBoundaryFace = lateralScvf.makeBoundaryFace(lateralStaggeredFaceCenter);
                    lateralFlux += problem.neumann(element, fvGeometry, elemVolVars, elemFaceVars, lateralBoundaryFace)[Indices::velocity(scvf.directionIndex())]
                                                  * elemVolVars[lateralScvf.insideScvIdx()].extrusionFactor() * lateralScvf.area() * 0.5;
                    continue;
                }

                // Handle wall-function fluxes (only required for RANS models)
                if (incorporateWallFunction_(lateralFlux, problem, element, fvGeometry, scvf, elemVolVars, elemFaceVars, localSubFaceIdx))
                    continue;

                // Check the consistency of the boundary conditions, only one of the following must be set
                std::bitset<3> admittableBcTypes;
                admittableBcTypes.set(0, lateralFaceBoundaryTypes->isDirichlet(Indices::pressureIdx));
                admittableBcTypes.set(1, lateralFaceBoundaryTypes->isBJS(Indices::velocity(scvf.directionIndex())));
                admittableBcTypes.set(2, lateralFaceBoundaryTypes->isDirichlet(Indices::velocity(scvf.directionIndex())));
                if (admittableBcTypes.count() != 1)
                {
                    DUNE_THROW(Dune::InvalidStateException,  "Something went wrong with the boundary conditions "
                    "for the momentum equations at global position " << lateralStaggeredFaceCenter_(scvf, localSubFaceIdx));
                }
            }

            // If there is no symmetry or Neumann boundary condition for the given sub face, proceed to calculate the tangential momentum flux.
            if (problem.enableInertiaTerms())
                lateralFlux += computeAdvectivePartOfLateralMomentumFlux_(problem, fvGeometry, element,
                                                                         scvf, elemVolVars, faceVars,
                                                                         gridFluxVarsCache, lateralFaceBoundaryTypes, localSubFaceIdx);

            lateralFlux += computeDiffusivePartOfLateralMomentumFlux_(problem, fvGeometry, element,
                                                                     scvf, elemVolVars, faceVars,
                                                                     lateralFaceBoundaryTypes, localSubFaceIdx);
        }
        return lateralFlux;
    }

    /*!
     * \brief Returns the momentum flux over an inflow or outflow boundary face.
     *
     * \verbatim
     *                    scvf      //
     *              ---------=======//               == and # staggered half-control-volume
     *              |      ||      #// current scvf
     *              |      ||      #//               # staggered boundary face over which fluxes are calculated
     *              |      ||      x~~~~> vel.Self
     *              |      ||      #//               x dof position
     *        scvf  |      ||      #//
     *              --------========//               -- element
     *                   scvf       //
     *                                              // boundary
     * \endverbatim
     */
    FacePrimaryVariables inflowOutflowBoundaryFlux(const Problem& problem,
                                                   const Element& element,
                                                   const SubControlVolumeFace& scvf,
                                                   const ElementVolumeVariables& elemVolVars,
                                                   const ElementFaceVariables& elemFaceVars) const
    {
        FacePrimaryVariables inOrOutflow(0.0);
        const auto& insideVolVars = elemVolVars[scvf.insideScvIdx()];

        // Advective momentum flux.
        if (problem.enableInertiaTerms())
        {
            const Scalar velocitySelf = elemFaceVars[scvf].velocitySelf();
            const auto& outsideVolVars = elemVolVars[scvf.outsideScvIdx()];
            const auto& upVolVars = (scvf.directionSign() == sign(velocitySelf)) ?
                                    insideVolVars : outsideVolVars;

            inOrOutflow += velocitySelf * velocitySelf * upVolVars.density();
        }

        // Apply a pressure at the boundary.
        const Scalar boundaryPressure = normalizePressure
                                        ? (problem.dirichlet(element, scvf)[Indices::pressureIdx] -
                                           problem.initial(scvf)[Indices::pressureIdx])
                                        : problem.dirichlet(element, scvf)[Indices::pressureIdx];
        inOrOutflow += boundaryPressure;

        // Account for the orientation of the face at the boundary,
        return inOrOutflow * scvf.directionSign() * scvf.area() * insideVolVars.extrusionFactor();
    }

private:

    /*!
     * \brief Returns the advective momentum flux over the staggered face perpendicular to the scvf
     *        where the velocity dof of interest lives (coinciding with the element's scvfs).
     *
     * \verbatim
     *              ----------------
     *              |              |
     *              |    transp.   |
     *              |      vel.    |~~~~> vel.Parallel
     *              |       ^      |
     *              |       |      |
     *       scvf   ---------#######                 || and # staggered half-control-volume
     *              |      ||      | current scvf
     *              |      ||      |                 # normal staggered faces over which fluxes are calculated
     *              |      ||      x~~~~> vel.Self
     *              |      ||      |                 x dof position
     *        scvf  |      ||      |
     *              ---------#######                -- elements
     *                 scvf
     * \endverbatim
     */
    FacePrimaryVariables computeAdvectivePartOfLateralMomentumFlux_(const Problem& problem,
                                                                    const FVElementGeometry& fvGeometry,
                                                                    const Element& element,
                                                                    const SubControlVolumeFace& scvf,
                                                                    const ElementVolumeVariables& elemVolVars,
                                                                    const FaceVariables& faceVars,
                                                                    const GridFluxVariablesCache& gridFluxVarsCache,
                                                                    const Dune::Std::optional<BoundaryTypes>& lateralFaceBoundaryTypes,
                                                                    const int localSubFaceIdx)
    {
        const auto eIdx = scvf.insideScvIdx();
        const auto& lateralFace = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);

        // Get the transporting velocity, located at the scvf perpendicular to the current scvf where the dof
        // of interest is located.
        const Scalar transportingVelocity = faceVars.velocityLateralInside(localSubFaceIdx);

        return StaggeredUpwindFluxVariables<TypeTag, upwindSchemeOrder>::computeUpwindedLateralMomentum(problem, fvGeometry, element, scvf, elemVolVars, faceVars,
                                                                     gridFluxVarsCache, localSubFaceIdx, lateralFaceBoundaryTypes)
               * transportingVelocity * lateralFace.directionSign() * lateralFace.area() * 0.5 * extrusionFactor_(elemVolVars, lateralFace);
    }

    /*!
     * \brief Returns the diffusive momentum flux over the staggered face perpendicular to the scvf
     *        where the velocity dof of interest lives (coinciding with the element's scvfs).
     *
     * \verbatim
     *              ----------------
     *              |              |vel.
     *              |    in.norm.  |Parallel
     *              |       vel.   |~~~~>
     *              |       ^      |        ^ out.norm.vel.
     *              |       |      |        |
     *       scvf   ---------#######:::::::::       || and # staggered half-control-volume (own element)
     *              |      ||      | curr. ::
     *              |      ||      | scvf  ::       :: staggered half-control-volume (neighbor element)
     *              |      ||      x~~~~>  ::
     *              |      ||      | vel.  ::       # lateral staggered faces over which fluxes are calculated
     *        scvf  |      ||      | Self  ::
     *              ---------#######:::::::::       x dof position
     *                 scvf
     *                                              -- elements
     * \endverbatim
     */
    FacePrimaryVariables computeDiffusivePartOfLateralMomentumFlux_(const Problem& problem,
                                                                    const FVElementGeometry& fvGeometry,
                                                                    const Element& element,
                                                                    const SubControlVolumeFace& scvf,
                                                                    const ElementVolumeVariables& elemVolVars,
                                                                    const FaceVariables& faceVars,
                                                                    const Dune::Std::optional<BoundaryTypes>& lateralFaceBoundaryTypes,
                                                                    const int localSubFaceIdx)
    {
        const auto eIdx = scvf.insideScvIdx();
        const auto& lateralFace = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);

        FacePrimaryVariables lateralDiffusiveFlux(0.0);

        static const bool enableUnsymmetrizedVelocityGradient
            = getParamFromGroup<bool>(problem.paramGroup(), "FreeFlow.EnableUnsymmetrizedVelocityGradient", false);

        // Get the volume variables of the own and the neighboring element. The neighboring
        // element is adjacent to the staggered face normal to the current scvf
        // where the dof of interest is located.
        const auto& insideVolVars = elemVolVars[lateralFace.insideScvIdx()];
        const auto& outsideVolVars = elemVolVars[lateralFace.outsideScvIdx()];

        // Get the averaged viscosity at the staggered face normal to the current scvf.
        const Scalar muAvg = lateralFace.boundary()
                             ? insideVolVars.effectiveViscosity()
                             : (insideVolVars.effectiveViscosity() + outsideVolVars.effectiveViscosity()) * 0.5;

        // Consider the shear stress caused by the gradient of the velocities normal to our face of interest.
        if (!enableUnsymmetrizedVelocityGradient)
        {
            // Create a boundaryTypes object (will be empty if not at a boundary).
            Dune::Std::optional<BoundaryTypes> bcTypes;

            // Get the boundary conditions if we are at a boundary. We sample the type of BC at the center of the current scvf.
            if (scvf.boundary())
                bcTypes.emplace(problem.boundaryTypes(element, scvf));

            // Check if we have to do the computation
            const bool enable = [&]()
            {
                // Always consider this term in the interior domain.
                if (!scvf.boundary())
                    return true;

                // If we are at a boundary and a Dirichlet BC for pressure is set, a gradient of zero is implictly assumed for all velocities,
                // thus no further calculations are required.
                if (bcTypes && bcTypes->isDirichlet(Indices::pressureIdx))
                    return false;

                // If we are at a boundary and neither a Dirichlet BC or a Beavers-Joseph slip condition for the tangential velocity is set,
                // we cannot calculate a velocity gradient and thus skip this part. TODO: is this the right approach?
                return (bcTypes && (bcTypes->isDirichlet(Indices::velocity(lateralFace.directionIndex())) ||
                                    bcTypes->isBJS(Indices::velocity(lateralFace.directionIndex()))));
            }();

            if (enable)
            {
                // For the velocityGrad_ji gradient, get the velocities perpendicular to the velocity at the current scvf.
                // The inner one is located at staggered face within the own element,
                // the outer one at the respective staggered face of the element on the other side of the
                // current scvf.
                const Scalar innerLateralVelocity = faceVars.velocityLateralInside(localSubFaceIdx);
                const Scalar outerLateralVelocity = [&]()
                {
                    if (!scvf.boundary())
                        return faceVars.velocityLateralOutside(localSubFaceIdx);
                    else if (bcTypes->isDirichlet(Indices::velocity(lateralFace.directionIndex())))
                    {
                        // Sample the value of the Dirichlet BC at the center of the lateral face intersecting with the boundary.
                        const auto& lateralBoundaryFacePos = lateralStaggeredFaceCenter_(scvf, localSubFaceIdx);
                        return problem.dirichlet(element, scvf.makeBoundaryFace(lateralBoundaryFacePos))[Indices::velocity(lateralFace.directionIndex())];
                    }
                    else
                    {
                        // Compute the BJS slip velocity at the boundary. Note that the relevant velocity gradient is now
                        // perpendicular to the own scvf.
                        const auto& scv = fvGeometry.scv(scvf.insideScvIdx());
                        return problem.bjsVelocity(element, scv, scvf, innerLateralVelocity);
                    }
                }();

                // Calculate the velocity gradient in positive coordinate direction.
                const Scalar lateralDeltaV = scvf.normalInPosCoordDir()
                                            ? (outerLateralVelocity - innerLateralVelocity)
                                            : (innerLateralVelocity - outerLateralVelocity);

                const Scalar velocityGrad_ji = lateralDeltaV / scvf.pairData(localSubFaceIdx).lateralDistance;

                // Account for the orientation of the staggered normal face's outer normal vector.
                lateralDiffusiveFlux -= muAvg * velocityGrad_ji * lateralFace.directionSign();
            }
        }

        // Consider the shear stress caused by the gradient of the velocities parallel to our face of interest.
        // If we have a Dirichlet condition for the pressure at the lateral face we assume to have a zero velocityGrad_ij velocity gradient
        // so we can skip the computation.
        if (!lateralFaceBoundaryTypes || !lateralFaceBoundaryTypes->isDirichlet(Indices::pressureIdx))
        {
            // For the velocityGrad_ij derivative, get the velocities at the current (own) scvf
            // and at the parallel one at the neighboring scvf.
            const Scalar innerParallelVelocity = faceVars.velocitySelf();

            const auto getParallelVelocity = [&]()
            {
                if (!lateralFace.boundary())
                    return faceVars.velocityParallel(localSubFaceIdx, 0);
                else if (lateralFaceBoundaryTypes->isDirichlet(Indices::velocity(scvf.directionIndex())))
                {
                    // Sample the value of the Dirichlet BC at the center of the staggered lateral face.
                    const auto& lateralBoundaryFacePos = lateralStaggeredFaceCenter_(scvf, localSubFaceIdx);
                    return problem.dirichlet(element, lateralFace.makeBoundaryFace(lateralBoundaryFacePos))[Indices::velocity(scvf.directionIndex())];
                }
                else
                {
                    const auto& scv = fvGeometry.scv(scvf.insideScvIdx());
                    return problem.bjsVelocity(element, scv, lateralFace, innerParallelVelocity);
                }
            };

            const Scalar outerParallelVelocity = getParallelVelocity();

            // The velocity gradient already accounts for the orientation
            // of the staggered face's outer normal vector. This also correctly accounts for the reduced
            // distance used in the gradient if the lateral scvf lies on a boundary.
            const Scalar velocityGrad_ij = (outerParallelVelocity - innerParallelVelocity)
                                          / scvf.parallelDofsDistance(localSubFaceIdx, 0);

            lateralDiffusiveFlux -= muAvg * velocityGrad_ij;
        }

        // Account for the area of the staggered lateral face (0.5 of the coinciding scfv).
        return lateralDiffusiveFlux * lateralFace.area() * 0.5 * extrusionFactor_(elemVolVars, lateralFace);
    }


    /*!
     * \brief Get the location of the lateral staggered face's center.
     *        Only needed for boundary conditions if the current scvf or the lateral one is on a bounary.
     *
     * \verbatim
     *      --------#######o                 || frontal face of staggered half-control-volume
     *      |      ||      | current scvf    #  lateral staggered face of interest (may lie on a boundary)
     *      |      ||      |                 x  dof position
     *      |      ||      x~~~~> vel.Self   -- element boundaries, current scvf may lie on a boundary
     *      |      ||      |                 o  position at which the boundary conditions will be evaluated
     *      |      ||      |                    (lateralStaggeredFaceCenter)
     *      ----------------
     * \endverbatim
     */
    const GlobalPosition& lateralStaggeredFaceCenter_(const SubControlVolumeFace& scvf, const int localSubFaceIdx) const
    {
        return scvf.pairData(localSubFaceIdx).lateralStaggeredFaceCenter;
    };

    //! helper function to get the averaged extrusion factor for a face
    static Scalar extrusionFactor_(const ElementVolumeVariables& elemVolVars, const SubControlVolumeFace& scvf)
    {
        const auto& insideVolVars = elemVolVars[scvf.insideScvIdx()];
        const auto& outsideVolVars = elemVolVars[scvf.outsideScvIdx()];
        return harmonicMean(insideVolVars.extrusionFactor(), outsideVolVars.extrusionFactor());
    }
    //! do nothing if no turbulence model is used
    template<class ...Args, bool turbulenceModel = ModelTraits::usesTurbulenceModel(), std::enable_if_t<!turbulenceModel, int> = 0>
    bool incorporateWallFunction_(Args&&... args) const
    { return false; }

    //! if a turbulence model is used, ask the problem is a wall function shall be employed and get the flux accordingly
    template<bool turbulenceModel = ModelTraits::usesTurbulenceModel(), std::enable_if_t<turbulenceModel, int> = 0>
    bool incorporateWallFunction_(FacePrimaryVariables& lateralFlux,
                                  const Problem& problem,
                                  const Element& element,
                                  const FVElementGeometry& fvGeometry,
                                  const SubControlVolumeFace& scvf,
                                  const ElementVolumeVariables& elemVolVars,
                                  const ElementFaceVariables& elemFaceVars,
                                  const std::size_t localSubFaceIdx) const
    {
        const auto eIdx = scvf.insideScvIdx();
        const auto& lateralScvf = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);

        if (problem.useWallFunction(element, lateralScvf, Indices::velocity(scvf.directionIndex())))
        {
            const auto& lateralStaggeredFaceCenter = lateralStaggeredFaceCenter_(scvf, localSubFaceIdx);
            const auto lateralBoundaryFace = lateralScvf.makeBoundaryFace(lateralStaggeredFaceCenter);
            lateralFlux += problem.wallFunction(element, fvGeometry, elemVolVars, elemFaceVars, scvf, lateralBoundaryFace)[Indices::velocity(scvf.directionIndex())]
                                               * extrusionFactor_(elemVolVars, lateralScvf) * lateralScvf.area() * 0.5;
            return true;
        }
        else
            return false;
    }
};

} // end namespace Dumux

#endif // DUMUX_NAVIERSTOKES_STAGGERED_FLUXVARIABLES_HH
