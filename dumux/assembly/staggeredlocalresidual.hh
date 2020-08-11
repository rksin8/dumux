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
 * \ingroup Assembly
 * \ingroup StaggeredDiscretization
 * \brief Calculates the element-wise residual for the staggered FV scheme
 */
#ifndef DUMUX_STAGGERED_LOCAL_RESIDUAL_HH
#define DUMUX_STAGGERED_LOCAL_RESIDUAL_HH

#include <dumux/common/timeloop.hh>
#include <dumux/common/properties.hh>
#include <dumux/discretization/extrusion.hh>

#include "simpleassemblystructs.hh"

namespace Dumux {

/*!
 * \ingroup Assembly
 * \ingroup StaggeredDiscretization
 * \brief Calculates the element-wise residual for the staggered FV scheme
 */
template<class TypeTag>
class StaggeredLocalResidual
{
    using GridView = typename GetPropType<TypeTag, Properties::GridGeometry>::GridView;

    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Implementation = GetPropType<TypeTag, Properties::LocalResidual>;
    using Problem = GetPropType<TypeTag, Properties::Problem>;
    using Element = typename GridView::template Codim<0>::Entity;
    using ElementBoundaryTypes = GetPropType<TypeTag, Properties::ElementBoundaryTypes>;
    using ElementVolumeVariables = typename GetPropType<TypeTag, Properties::GridVolumeVariables>::LocalView;
    using ElementFluxVariablesCache = typename GetPropType<TypeTag, Properties::GridFluxVariablesCache>::LocalView;
    using GridGeometry = GetPropType<TypeTag, Properties::GridGeometry>;
    using FVElementGeometry = typename GridGeometry::LocalView;
    using SubControlVolume = typename GridGeometry::SubControlVolume;
    using SubControlVolumeFace = typename GridGeometry::SubControlVolumeFace;
    using FaceSubControlVolume = typename GridGeometry::Traits::FaceSubControlVolume;
    using Extrusion = Extrusion_t<GridGeometry>;
    using CellCenterPrimaryVariables = GetPropType<TypeTag, Properties::CellCenterPrimaryVariables>;

    using CellCenterResidual = GetPropType<TypeTag, Properties::CellCenterPrimaryVariables>;
    using FaceResidual = GetPropType<TypeTag, Properties::FacePrimaryVariables>;
    using ElementFaceVariables = typename GetPropType<TypeTag, Properties::GridFaceVariables>::LocalView;

    using TimeLoop = TimeLoopBase<Scalar>;

    using Indices = typename GetPropType<TypeTag, Properties::ModelTraits>::Indices;

public:
    using CellCenterResidualValue = GetPropType<TypeTag, Properties::CellCenterPrimaryVariables>;
    using FaceResidualValue = GetPropType<TypeTag, Properties::FacePrimaryVariables>;
    using ElementResidualVector = CellCenterResidualValue;

    using SimpleMassBalanceSummands = GetPropType<TypeTag, Properties::SimpleMassBalanceSummands>;
    using SimpleMomentumBalanceSummands = GetPropType<TypeTag, Properties::SimpleMomentumBalanceSummands>;

    //! the constructor
    StaggeredLocalResidual(const Problem* problem,
                           const TimeLoop* timeLoop = nullptr)
    : problem_(problem)
    , timeLoop_(timeLoop)
    {}

    //! Convenience function to evaluate the flux and source terms for the cell center residual
    void evalFluxAndSourceForCellCenter(const Element& element,
                                                           const FVElementGeometry& fvGeometry,
                                                           const ElementVolumeVariables& elemVolVars,
                                                           const ElementFaceVariables& elemFaceVars,
                                                           const ElementBoundaryTypes& bcTypes,
                                                           const ElementFluxVariablesCache& elemFluxVarsCache,
                                                           SimpleMassBalanceSummands& simpleMassBalanceSummands) const
    {
        simpleMassBalanceSummands.setToZero(element, fvGeometry);

        // evaluate the source term
        for (auto&& scv : scvs(fvGeometry))
            asImp().evalSourceForCellCenter(simpleMassBalanceSummands, this->problem(), element, fvGeometry, elemVolVars, elemFaceVars, scv);

        // evaluate the flux term
        for (auto&& scvf : scvfs(fvGeometry))
            asImp().evalFluxForCellCenter(simpleMassBalanceSummands, this->problem(), element, fvGeometry, elemVolVars, elemFaceVars, bcTypes, elemFluxVarsCache, scvf);
    }

    //! Evaluate the flux terms for a cell center residual
    void evalFluxForCellCenter(SimpleMassBalanceSummands& simpleMassBalanceSummands,
                               const Problem& problem,
                               const Element& element,
                               const FVElementGeometry& fvGeometry,
                               const ElementVolumeVariables& elemVolVars,
                               const ElementFaceVariables& elemFaceVars,
                               const ElementBoundaryTypes& elemBcTypes,
                               const ElementFluxVariablesCache& elemFluxVarsCache,
                               const SubControlVolumeFace& scvf) const
    {
        if (!scvf.boundary())
             simpleMassBalanceSummands.coefficients[scvf.localFaceIdx()] += asImp_().computeFluxForCellCenter(problem, element, fvGeometry, elemVolVars, elemFaceVars, scvf, elemFluxVarsCache);
        else
            asImp_().computeBoundaryFluxForCellCenter(problem, element, fvGeometry, scvf, elemVolVars, elemFaceVars, elemBcTypes, elemFluxVarsCache, simpleMassBalanceSummands);
    }

    //! Evaluate the source terms for a cell center residual
    void evalSourceForCellCenter(SimpleMassBalanceSummands& simpleMassBalanceSummands,
                                 const Problem& problem,
                                 const Element& element,
                                 const FVElementGeometry& fvGeometry,
                                 const ElementVolumeVariables& curElemVolVars,
                                 const ElementFaceVariables& curElemFaceVars,
                                 const SubControlVolume& scv) const
    {
            const auto curExtrusionFactor = curElemVolVars[scv].extrusionFactor();

            // subtract the source term from the local rate
            auto source = asImp_().computeSourceForCellCenter(problem, element, fvGeometry, curElemVolVars, curElemFaceVars, scv);
            source *= Extrusion::volume(scv)*curExtrusionFactor;
            simpleMassBalanceSummands.RHS += source;
    }

    //! Evaluate the storage terms for a cell center residual
    void evalStorageForCellCenter(const Element &element,
                                                     const FVElementGeometry& fvGeometry,
                                                     const ElementVolumeVariables& prevElemVolVars,
                                                     const ElementVolumeVariables& curElemVolVars,
                                  SimpleMassBalanceSummands& simpleMassBalanceSummands) const
    {
        assert(timeLoop_ && "no time loop set for storage term evaluation");
        CellCenterResidualValue storage(0.0);

        for (auto&& scv : scvs(fvGeometry))
            asImp().evalStorageForCellCenter(simpleMassBalanceSummands, problem(), element, fvGeometry, prevElemVolVars, curElemVolVars, scv);
    }

    //! Evaluate the storage terms for a cell center residual
    void evalStorageForCellCenter(SimpleMassBalanceSummands& simpleMassBalanceSummands,
                                  const Problem& problem,
                                  const Element &element,
                                  const FVElementGeometry& fvGeometry,
                                  const ElementVolumeVariables& prevElemVolVars,
                                  const ElementVolumeVariables& curElemVolVars,
                                  const SubControlVolume& scv) const
    {
        CellCenterResidualValue storage(0.0);
        const auto& curVolVars = curElemVolVars[scv];
        const auto& prevVolVars = prevElemVolVars[scv];

        // mass balance within the element. this is the
        // \f$\frac{m}{\partial t}\f$ term if using implicit
        // euler as time discretization.

        // We might need a more explicit way for
        // doing the time discretization...
        auto prevCCStorage = asImp_().computeStorageForCellCenter(problem, scv, prevVolVars);
        auto curCCStorage = asImp_().computeStorageForCellCenter(problem, scv, curVolVars);

        prevCCStorage *= prevVolVars.extrusionFactor();
        curCCStorage *= curVolVars.extrusionFactor();

        storage = std::move(curCCStorage);
        storage -= std::move(prevCCStorage);
        storage *= Extrusion::volume(scv);
        storage /= timeLoop_->timeStepSize();

        simpleMassBalanceSummands.RHS -= storage;
    }

    //! for compatibility with FVLocalAssemblerBase
    template<class... Args>
    void evalFluxAndSource(Args&&... args) const
    {
        return CellCenterResidualValue(0.0);
    }

    //! for compatibility with FVLocalAssemblerBase
    template<class... Args>
    void evalStorage(Args&&... args) const
    {
        return CellCenterResidualValue(0.0);
    }

    /*!
     * \name User interface
     * \note The following methods are usually expensive to evaluate
     *       They are useful for outputting residual information.
     */
    // \{

    //! Convenience function to evaluate the flux and source terms for the face residual
    void evalFluxAndSourceForFace(const Element& element,
                                               const FVElementGeometry& fvGeometry,
                                               const ElementVolumeVariables& elemVolVars,
                                               const ElementFaceVariables& elemFaceVars,
                                               const ElementBoundaryTypes& bcTypes,
                                               const ElementFluxVariablesCache& elemFluxVarsCache,
                                               const SubControlVolumeFace& scvf,
                                  SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands) const
    {
        simpleMomentumBalanceSummands.setToZero(scvf);

        asImp().evalSourceForFace(simpleMomentumBalanceSummands, this->problem(), element, fvGeometry, elemVolVars, elemFaceVars, scvf);
        asImp().evalFluxForFace(simpleMomentumBalanceSummands, this->problem(), element, fvGeometry, elemVolVars, elemFaceVars, bcTypes, elemFluxVarsCache, scvf);
    }

    //! Evaluate the flux terms for a face residual
    void evalFluxForFace(SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands,
                         const Problem& problem,
                         const Element& element,
                         const FVElementGeometry& fvGeometry,
                         const ElementVolumeVariables& elemVolVars,
                         const ElementFaceVariables& elemFaceVars,
                         const ElementBoundaryTypes& elemBcTypes,
                         const ElementFluxVariablesCache& elemFluxVarsCache,
                         const SubControlVolumeFace& scvf) const
    {
        if (!scvf.boundary())
            asImp_().computeFluxForFace(problem, element, scvf, fvGeometry, elemVolVars, elemFaceVars, elemFluxVarsCache, simpleMomentumBalanceSummands);
        else
            asImp_().computeBoundaryFluxForFace(problem, element, fvGeometry, scvf, elemVolVars, elemFaceVars, elemBcTypes, elemFluxVarsCache, simpleMomentumBalanceSummands);
    }

    //! Evaluate the source terms for a face residual
    void evalSourceForFace(SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands,
                           const Problem& problem,
                           const Element& element,
                           const FVElementGeometry& fvGeometry,
                           const ElementVolumeVariables& elemVolVars,
                           const ElementFaceVariables& elemFaceVars,
                           const SubControlVolumeFace& scvf) const
    {
        // the source term:
        auto source = asImp_().computeSourceForFace(problem, element, fvGeometry, scvf, elemVolVars, elemFaceVars);
        const auto& scv = fvGeometry.scv(scvf.insideScvIdx());
        const auto extrusionFactor = elemVolVars[scv].extrusionFactor();

        // construct staggered scv (half of the element)
        auto faceScvCenter = scvf.center() + scv.center();
        faceScvCenter *= 0.5;
        FaceSubControlVolume faceScv(faceScvCenter, 0.5*scv.volume());

        source *= Extrusion::volume(faceScv)*extrusionFactor;
        simpleMomentumBalanceSummands.RHS += source;
    }

    //! Evaluate the storage terms for a face residual
    void evalStorageForFace(const Element& element,
                                         const FVElementGeometry& fvGeometry,
                                         const ElementVolumeVariables& prevElemVolVars,
                                         const ElementVolumeVariables& curElemVolVars,
                                         const ElementFaceVariables& prevElemFaceVars,
                                         const ElementFaceVariables& curElemFaceVars,
                                         const SubControlVolumeFace& scvf,
                           SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands) const
    {
        assert(timeLoop_ && "no time loop set for storage term evaluation");
        FaceResidualValue storage(0.0);
        asImp().evalStorageForFace(simpleMomentumBalanceSummands, problem(), element, fvGeometry, prevElemVolVars, curElemVolVars, prevElemFaceVars, curElemFaceVars, scvf);
    }

    //! Evaluate the storage terms for a face residual
    void evalStorageForFace(SimpleMomentumBalanceSummands& simpleMomentumBalanceSummands,
                            const Problem& problem,
                            const Element& element,
                            const FVElementGeometry& fvGeometry,
                            const ElementVolumeVariables& prevElemVolVars,
                            const ElementVolumeVariables& curElemVolVars,
                            const ElementFaceVariables& prevElemFaceVars,
                            const ElementFaceVariables& curElemFaceVars,
                            const SubControlVolumeFace& scvf) const
    {
        const auto& scv = fvGeometry.scv(scvf.insideScvIdx());

        // construct staggered scv (half of the element)
        auto faceScvCenter = scvf.center() + scv.center();
        faceScvCenter *= 0.5;
        FaceSubControlVolume faceScv(faceScvCenter, 0.5*scv.volume());

        auto prevFaceStorage = prevElemVolVars[scv].density() * prevElemFaceVars[scvf].velocitySelf();
        const auto extrusionFactor = curElemVolVars[scv].extrusionFactor();

        auto RHSContribution = std::move(prevFaceStorage);
        // multiply by 0.5 because we only consider half of a staggered control volume here
        RHSContribution *= Extrusion::volume(faceScv)*extrusionFactor;
        RHSContribution /= timeLoop_->timeStepSize();

        simpleMomentumBalanceSummands.RHS += RHSContribution;

        //compute contribution to selfCoefficient
        if(scvf.boundary() && problem.boundaryTypes(element, scvf).isDirichlet(Indices::velocity(scvf.directionIndex()))){
            auto curFaceStorage = - curElemVolVars[scv].density() * curElemFaceVars[scvf].velocitySelf();

            auto secondRHSContribution = std::move(curFaceStorage);
            secondRHSContribution *= Extrusion::volume(faceScv)*extrusionFactor;
            secondRHSContribution /= timeLoop_->timeStepSize();

            simpleMomentumBalanceSummands.RHS += secondRHSContribution;
        }
        else {
            auto curFaceStorage = curElemVolVars[scv].density();

            auto selfCoefficientContribution = std::move(curFaceStorage);
            selfCoefficientContribution *= Extrusion::volume(faceScv)*extrusionFactor;
            selfCoefficientContribution /= timeLoop_->timeStepSize();
            simpleMomentumBalanceSummands.selfCoefficient += selfCoefficientContribution;
        }
    }

    //! If no solution has been set, we treat the problem as stationary.
    bool isStationary() const
    { return !timeLoop_; }

    //! the problem
    const Problem& problem() const
    { return *problem_; }

protected:

    Implementation &asImp_()
    { return *static_cast<Implementation*>(this); }

    const Implementation &asImp_() const
    { return *static_cast<const Implementation*>(this); }


    TimeLoop& timeLoop()
    { return *timeLoop_; }

    const TimeLoop& timeLoop() const
    { return *timeLoop_; }

    Implementation &asImp()
    { return *static_cast<Implementation*>(this); }

    const Implementation &asImp() const
    { return *static_cast<const Implementation*>(this); }

private:
    const Problem* problem_; //!< the problem we are assembling this residual for
    const TimeLoop* timeLoop_;

};

} // end namespace Dumux

#endif   // DUMUX_CC_LOCAL_RESIDUAL_HH
