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
 * \copydoc Dumux::StaggeredVelocityGradients
 */
#ifndef DUMUX_NAVIERSTOKES_STAGGERED_VELOCITYGRADIENTS_HH
#define DUMUX_NAVIERSTOKES_STAGGERED_VELOCITYGRADIENTS_HH

#include <optional>
#include <dumux/common/exceptions.hh>
#include <dumux/common/parameters.hh>

namespace Dumux {

/*!
 * \ingroup NavierStokesModel
 * \brief Helper class for calculating the velocity gradients for the Navier-Stokes model using the staggered grid discretization.
 */
template<class Scalar, class GridGeometry, class BoundaryTypes, class Indices>
class StaggeredVelocityGradients
{
    using FVElementGeometry = typename GridGeometry::LocalView;
    using GridView = typename GridGeometry::GridView;
    using Element = typename GridView::template Codim<0>::Entity;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;
    using GlobalPosition = typename Element::Geometry::GlobalCoordinate;

public:

    /*!
     * \brief Returns the in-axis velocity gradient.
     *
     * \verbatim
     *              ---------=======                 == and # staggered half-control-volume
     *              |       #      | current scvf
     *              |       #      |                 # staggered face over which fluxes are calculated
     *   vel.Opp <~~|       O~~>   x~~~~> vel.Self
     *              |       #      |                 x dof position
     *        scvf  |       #      |
     *              --------========                 -- element
     *
     *                                               O position at which gradient is evaluated
     * \endverbatim
     */
    template<class Problem, class FaceVariables, class SimpleMomentumBalanceSummands>
    void velocityGradII(const Problem& problem,
                        const Element& element,
                                 const SubControlVolumeFace& scvf,
                        const FVElementGeometry& fvGeometry,
                                 const FaceVariables& faceVars,
                                 SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands,
                                 Scalar factor)
    {
        if(scvf.boundary() && problem.boundaryTypes(element, scvf).isDirichlet(Indices::velocity(scvf.directionIndex()))){
            simpleMomentumBalanceSummands.RHS -= faceVars.velocitySelf() * factor;;
        }
        else {
            simpleMomentumBalanceSummands.selfCoefficient += factor;;
        }

        const auto eIdx = scvf.insideScvIdx();
        const auto opposingFace = fvGeometry.scvf(eIdx, scvf.localIdxOpposingFace());

        if(opposingFace.boundary() && problem.boundaryTypes(element, opposingFace).isDirichlet(Indices::velocity(scvf.directionIndex()))){
            simpleMomentumBalanceSummands.RHS += faceVars.velocityOpposite()*factor;
        }
        else {
            simpleMomentumBalanceSummands.oppositeCoefficient -= factor;
        }
    }

    /*!
     * \brief Returns the velocity gradient perpendicular to the orientation of our current scvf.
     *
     * \verbatim
     *              ----------------
     *              |              |vel.
     *              |              |Parallel
     *              |              |~~~~>       ------->
     *              |              |             ------>     * gradient
     *              |              |              ----->
     *       scvf   ---------######O:::::::::      ---->     || and # staggered half-control-volume (own element)
     *              |      ||      | curr. ::       --->
     *              |      ||      | scvf  ::        -->     :: staggered half-control-volume (neighbor element)
     *              |      ||      x~~~~>  ::         ->
     *              |      ||      | vel.  ::                # lateral staggered faces over which fluxes are calculated
     *        scvf  |      ||      | Self  ::
     *              ---------#######:::::::::                x dof position
     *                 scvf
     *                                                       -- elements
     *
     *                                                       O position at which gradient is evaluated
     * \endverbatim
     */
    template<class Problem, class FaceVariables, class SimpleMomentumBalanceSummands>
    void velocityGradIJ(const Problem& problem,
                                 const Element& element,
                                 const FVElementGeometry& fvGeometry,
                                 const SubControlVolumeFace& scvf,
                                 const FaceVariables& faceVars,
                                 const std::optional<BoundaryTypes>& currentScvfBoundaryTypes,
                                 const std::optional<BoundaryTypes>& lateralFaceBoundaryTypes,
                                 const std::size_t localSubFaceIdx,
                                 SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands,
                                 Scalar factor)
    {
        const auto eIdx = scvf.insideScvIdx();
        const auto& lateralScvf = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);

        factor *= lateralScvf.directionSign();
        factor /= scvf.parallelDofsDistance(localSubFaceIdx, 0);

        if (!(lateralScvf.boundary() && problem.boundaryTypes(element, lateralScvf).isOutflow(Indices::velocity(scvf.directionIndex()))))
        {
            const Scalar innerParallelVelocity = faceVars.velocitySelf();
            simpleMomentumBalanceSummands.RHS -= factor* innerParallelVelocity;
        }
        else
        {
           simpleMomentumBalanceSummands.selfCoefficient += factor;
        }

        if (!lateralScvf.boundary())
        {
            const auto parallelFace = fvGeometry.scvf(lateralScvf.outsideScvIdx(), scvf.localFaceIdx());
            if (parallelFace.boundary() && problem.boundaryTypes(element, parallelFace).isDirichlet(Indices::velocity(scvf.directionIndex()))){
                simpleMomentumBalanceSummands.RHS += factor * faceVars.velocityParallel(localSubFaceIdx, 0);
            }
            else{
                simpleMomentumBalanceSummands.parallelCoefficients[localSubFaceIdx] -= factor;
            }
        }
        else if (lateralFaceBoundaryTypes->isDirichlet(Indices::velocity(scvf.directionIndex())))
        {
            // Sample the value of the Dirichlet BC at the center of the staggered lateral face.
            const auto& lateralBoundaryFacePos = lateralStaggeredFaceCenter_(scvf, localSubFaceIdx);
            const auto lateralBoundaryFace = makeStaggeredBoundaryFace(lateralScvf, lateralBoundaryFacePos);
            simpleMomentumBalanceSummands.RHS += factor * problem.dirichlet(element, lateralBoundaryFace)[Indices::velocity(scvf.directionIndex())];
        }
        else
        {
            DUNE_THROW(Dune::InvalidStateException, "SIMPLE not prepared for other boundary types");
        }
    }

    /*!
     * \brief Returns the velocity gradient in line with our current scvf.
     *
     * \verbatim
     *                      ^       gradient
     *                      |  ^
     *                      |  |  ^
     *                      |  |  |  ^
     *                      |  |  |  |  ^
     *                      |  |  |  |  |  ^
     *                      |  |  |  |  |  |
     *
     *              ----------------
     *              |              |
     *              |    in.norm.  |
     *              |       vel.   |
     *              |       ^      |        ^ out.norm.vel.
     *              |       |      |        |
     *       scvf   ---------######O:::::::::       || and # staggered half-control-volume (own element)
     *              |      ||      | curr. ::
     *              |      ||      | scvf  ::       :: staggered half-control-volume (neighbor element)
     *              |      ||      x~~~~>  ::
     *              |      ||      | vel.  ::       # lateral staggered faces over which fluxes are calculated
     *        scvf  |      ||      | Self  ::
     *              ---------#######:::::::::       x dof position
     *                 scvf
     *                                              -- elements
     *
     *                                              O position at which gradient is evaluated
     * \endverbatim
     */
    template<class Problem, class FaceVariables, class SimpleMomentumBalanceSummands>
    void velocityGradJI(const Problem& problem,
                                 const Element& element,
                                 const FVElementGeometry& fvGeometry,
                                 const SubControlVolumeFace& scvf,
                                 const FaceVariables& faceVars,
                                 const std::optional<BoundaryTypes>& currentScvfBoundaryTypes,
                                 const std::optional<BoundaryTypes>& lateralFaceBoundaryTypes,
                                 const std::size_t localSubFaceIdx,
                                 SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands,
                                 Scalar factor)
    {
        // For the velocityGrad_ji gradient, get the velocities perpendicular to the velocity at the current scvf.
        // The inner one is located at staggered face within the own element,
        // the outer one at the respective staggered face of the element on the other side of the
        // current scvf.

        const auto eIdx = scvf.insideScvIdx();
        const auto& lateralScvf = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);

        // Assume a zero velocity gradient for pressure boundary conditions.
        if (currentScvfBoundaryTypes && currentScvfBoundaryTypes->isDirichlet(Indices::pressureIdx))
            return;

        factor /= scvf.pairData(localSubFaceIdx).lateralDistance;
        if (scvf.normalInPosCoordDir())
            factor *= -1;

        if (lateralScvf.boundary() && problem.boundaryTypes(element, lateralScvf).isDirichlet(Indices::velocity(scvf.directionIndex()))){
            const Scalar innerLateralVelocity = faceVars.velocityLateralInside(localSubFaceIdx);
            simpleMomentumBalanceSummands.RHS += factor * innerLateralVelocity;
        }
        else{
            simpleMomentumBalanceSummands.innerNormalCoefficients[localSubFaceIdx] -= factor;
        }

        const auto outerNormalFace = fvGeometry.scvf(scvf.outsideScvIdx(), lateralScvf.localFaceIdx());

        if (!scvf.boundary())
        {
            if(outerNormalFace.boundary() && problem.boundaryTypes(element, outerNormalFace).isDirichlet(Indices::velocity(scvf.directionIndex()))){
                simpleMomentumBalanceSummands.RHS -= factor * faceVars.velocityLateralOutside(localSubFaceIdx);
            }
            else{
                simpleMomentumBalanceSummands.outerNormalCoefficients[localSubFaceIdx] += factor;
            }
        }
        else if (currentScvfBoundaryTypes->isDirichlet(Indices::velocity(lateralScvf.directionIndex())))
        {
            // Sample the value of the Dirichlet BC at the center of the lateral face intersecting with the boundary.
            const auto& lateralBoundaryFacePos = lateralStaggeredFaceCenter_(scvf, localSubFaceIdx);
            const auto lateralBoundaryFace = makeStaggeredBoundaryFace(scvf, lateralBoundaryFacePos);
            simpleMomentumBalanceSummands.RHS -= factor * problem.dirichlet(element, lateralBoundaryFace)[Indices::velocity(lateralScvf.directionIndex())];
        }
        else
        {
            DUNE_THROW(Dune::InvalidStateException, "SIMPLE not prepared for this boundary type ");
        }
    }

    /*!
     * \brief Returns the Beavers-Jospeh slip velocity for a scvf which lies on the boundary itself.
     *
     * \verbatim
     *                  in.norm.  B-J slip
     *                     vel.   vel.
     *                     ^       ^
     *                     |       |
     *       scvf   ---------######|*               * boundary
     *              |      ||      |* curr.
     *              |      ||      |* scvf          || and # staggered half-control-volume (own element)
     *              |      ||      x~~~~>
     *              |      ||      |* vel.          # lateral staggered faces
     *        scvf  |      ||      |* Self
     *              ---------#######*                x dof position
     *                 scvf
     *                                              -- element
     * \endverbatim
     *
     */
    template<class Problem, class FaceVariables>
    static Scalar beaversJosephVelocityAtCurrentScvf(const Problem& problem,
                                                     const Element& element,
                                                     const FVElementGeometry& fvGeometry,
                                                     const SubControlVolumeFace& scvf,
                                                     const FaceVariables& faceVars,
                                                     const std::optional<BoundaryTypes>& currentScvfBoundaryTypes,
                                                     const std::optional<BoundaryTypes>& lateralFaceBoundaryTypes,
                                                     const std::size_t localSubFaceIdx)
    {
        const auto eIdx = scvf.insideScvIdx();
        const auto& lateralScvf = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);
        const Scalar innerLateralVelocity = faceVars.velocityLateralInside(localSubFaceIdx);

        const auto tangentialVelocityGradient = [&]()
        {
            // If the current scvf is on a boundary and if a Dirichlet BC for the pressure or a BJ condition for
            // the slip velocity is set there, assume a tangential velocity gradient of zero along the lateral face
            // (towards the current scvf).
            static const bool unsymmetrizedGradientForBJ = getParamFromGroup<bool>(problem.paramGroup(),
                                                           "FreeFlow.EnableUnsymmetrizedVelocityGradientForBeaversJoseph", false);

            if (unsymmetrizedGradientForBJ)
                return 0.0;

            if (lateralScvf.boundary())
            {
                if (lateralFaceBoundaryTypes->isDirichlet(Indices::pressureIdx) ||
                    lateralFaceBoundaryTypes->isBeaversJoseph(Indices::velocity(scvf.directionIndex())))
                    return 0.0;
            }

            DUNE_THROW(Dune::InvalidStateException, "not correct here for SIMPLE.");
            return 0.0;
        }();

        return problem.beaversJosephVelocity(element,
                                             fvGeometry.scv(scvf.insideScvIdx()),
                                             lateralScvf,
                                             scvf, /*on boundary*/
                                             innerLateralVelocity,
                                             tangentialVelocityGradient);
    }

    /*!
     * \brief Returns the Beavers-Jospeh slip velocity for a lateral scvf which lies on the boundary.
     *
     * \verbatim
     *                             B-J slip                  * boundary
     *              ************** vel. *****
     *       scvf   ---------##### ~~~~> ::::                || and # staggered half-control-volume (own element)
     *              |      ||      | curr. ::
     *              |      ||      | scvf  ::                :: staggered half-control-volume (neighbor element)
     *              |      ||      x~~~~>  ::
     *              |      ||      | vel.  ::                # lateral staggered faces
     *        scvf  |      ||      | Self  ::
     *              ---------#######:::::::::                x dof position
     *                 scvf
     *                                                       -- elements
     * \endverbatim
     */
    template<class Problem, class FaceVariables>
    static Scalar beaversJosephVelocityAtLateralScvf(const Problem& problem,
                                                     const Element& element,
                                                     const FVElementGeometry& fvGeometry,
                                                     const SubControlVolumeFace& scvf,
                                                     const FaceVariables& faceVars,
                                                     const std::optional<BoundaryTypes>& currentScvfBoundaryTypes,
                                                     const std::optional<BoundaryTypes>& lateralFaceBoundaryTypes,
                                                     const std::size_t localSubFaceIdx)
    {
        const auto eIdx = scvf.insideScvIdx();
        const auto& lateralScvf = fvGeometry.scvf(eIdx, scvf.pairData(localSubFaceIdx).localLateralFaceIdx);
        const Scalar innerParallelVelocity = faceVars.velocitySelf();

        const auto tangentialVelocityGradient = [&]()
        {
            // If the current scvf is on a boundary and if a Dirichlet BC for the pressure or a BJ condition for
            // the slip velocity is set there, assume a tangential velocity gradient of zero along the lateral face
            // (towards the current scvf).
            static const bool unsymmetrizedGradientForBJ = getParamFromGroup<bool>(problem.paramGroup(),
                                                           "FreeFlow.EnableUnsymmetrizedVelocityGradientForBeaversJoseph", false);

            if (unsymmetrizedGradientForBJ)
                return 0.0;

            if (scvf.boundary())
            {
                if (currentScvfBoundaryTypes->isDirichlet(Indices::pressureIdx) ||
                    currentScvfBoundaryTypes->isBeaversJoseph(Indices::velocity(lateralScvf.directionIndex())))
                    return 0.0;
            }

            DUNE_THROW(Dune::InvalidStateException, "not correct here for SIMPLE.");
            return 0.0;
        }();

        return problem.beaversJosephVelocity(element,
                                             fvGeometry.scv(scvf.insideScvIdx()),
                                             scvf,
                                             lateralScvf, /*on boundary*/
                                             innerParallelVelocity,
                                             tangentialVelocityGradient);
    }

private:

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
    static const GlobalPosition& lateralStaggeredFaceCenter_(const SubControlVolumeFace& scvf, const int localSubFaceIdx)
    {
        return scvf.pairData(localSubFaceIdx).lateralStaggeredFaceCenter;
    };
};

} // end namespace Dumux

#endif // DUMUX_NAVIERSTOKES_STAGGERED_VELOCITYGRADIENTS_HH
