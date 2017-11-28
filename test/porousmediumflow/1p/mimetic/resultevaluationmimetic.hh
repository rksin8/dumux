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


#ifndef DUMUX_RESULTEVALUATION_MIMETIC_HH
#define DUMUX_RESULTEVALUATION_MIMETIC_HH

#include <dumux/porousmediumflow/1p/mimetic/model.hh>

namespace Dumux
{

/*!
 * \brief calculate errors for general 1p tests
 */
template <class TypeTag>
struct ResultEvaluation
{
    using GridView = typename GET_PROP_TYPE(TypeTag, GridView);
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);

    // Grid and world dimension
    static const int dim = GridView::dimension;
    static const int dimWorld = GridView::dimensionworld;

    using Indices = typename GET_PROP_TYPE(TypeTag, Indices);
    enum {
     // indices of the primary variables
     conti0EqIdx = Indices::conti0EqIdx,
     pressureIdx = Indices::pressureIdx,
     facePressureIdx = Indices::facePressureIdx
    };

    using PrimaryVariables = typename GET_PROP_TYPE(TypeTag, PrimaryVariables);
    using BoundaryTypes = typename GET_PROP_TYPE(TypeTag, BoundaryTypes);
    using TimeManager = typename GET_PROP_TYPE(TypeTag, TimeManager);
    using Element = typename GridView::template Codim<0>::Entity;
    using FVElementGeometry = typename GET_PROP_TYPE(TypeTag, FVElementGeometry);
    using SubControlVolume = typename GET_PROP_TYPE(TypeTag, SubControlVolume);
    using GlobalPosition = Dune::FieldVector<Scalar, dimWorld>;

    using FluxVariables = typename GET_PROP_TYPE(TypeTag, FluxVariables);
    using ElementFluxVariablesCache = typename GET_PROP_TYPE(TypeTag, ElementFluxVariablesCache);

    enum { isBox = GET_PROP_VALUE(TypeTag, ImplicitIsBox) };

    typedef typename GridView::Intersection Intersection;
    typedef Dune::BlockVector<Dune::FieldVector<Scalar, 1> > SolVector;

    using DofTypeIndices = typename GET_PROP(TypeTag, DofTypeIndices);
    typename DofTypeIndices::CellCenterIdx cellCenterIdx;
    typename DofTypeIndices::FaceIdx faceIdx;

private:

public:
    Scalar absL2Error;
    Scalar absH1ErrorApproxMin;
    Scalar absH1ErrorDiffMin;
    Scalar absH1Error;
    Scalar bilinearFormApprox;
    Scalar bilinearFormDiffApprox;
    Scalar residualTermApprox;
    Scalar residualTermDiff;
    Scalar residualTermError;
    Scalar relativeL2Error;
    Scalar relativeL2ErrorIn;
    Scalar relativeL2ErrorOut;
    Scalar relativeL2ErrorFaceFluxes;
    Scalar relativeL2ErrorFaceFluxesBound;
    Scalar relativeL2ErrorFaceFluxesInner;
    Scalar relVolL2Error;
    Scalar relVelL2Error;
    Scalar relVolVelL2Error;
    Scalar absL2ErrorFaceFluxes;
    Scalar uMinExact;
    Scalar uMaxExact;
    Scalar uMin;
    Scalar uMax;
    Scalar hMax;
    Scalar hMin;
    SolVector localRelativeFaceVelError;
    SolVector localRelativeFaceFluxDiff;

    template<class Problem>
    void evaluate(Problem& problem)
    {
        int numDofs = problem.gridView().size(0);
        localRelativeFaceVelError.resize(numDofs);
        localRelativeFaceFluxDiff.resize(numDofs);

        uMinExact = 1e100;
        uMaxExact = -1e100;
        uMin = 1e100;
        uMax = -1e100;
        hMax = 0.0;
        hMin = 1.0e100;

        absH1ErrorApproxMin = 0.0;
        absH1ErrorDiffMin = 0.0;
        absH1Error = 0.0;
        bilinearFormApprox = 0.0;
        bilinearFormDiffApprox = 0.0;
        residualTermApprox = 0.0;
        residualTermDiff = 0.0;
        residualTermError = 0.0;

        Scalar numerator = 0;
        Scalar denominator = 0;

        Scalar numeratorFaceFlux = 0;
        Scalar denominatorFaceFlux = 0;
        Scalar numeratorFaceFluxBound = 0;
        Scalar denominatorFaceFluxBound = 0;
        Scalar numeratorFaceFluxInner = 0;
        Scalar denominatorFaceFluxInner = 0;

        Scalar numeratorFaceVel = 0.0;
        Scalar denominatorFaceFlux_relVol = 0;
        Scalar totalFaceVol = 0.0;
        absL2ErrorFaceFluxes = 0.0;
        std::vector<bool> alreadyVisited(numDofs, false);

        Scalar totalVelSum = 0.0;

        for (const auto& element : elements(problem.gridView()))
        {
            int eIdx = problem.elementMapper().index(element);
            auto fvGeometry = localView(problem.model().globalFvGeometry());
            fvGeometry.bindElement(element);

            auto scv = fvGeometry.scv(eIdx);
            //volume of element
            Scalar volume = scv.volume();
            auto cellCenter = scv.center();

            Scalar exactPressure = problem.exact(cellCenter);
            Scalar approxPressure = problem.model().curSol()[cellCenterIdx][eIdx][pressureIdx];

            localRelativeFaceVelError[eIdx] = 0.0;
            localRelativeFaceFluxDiff[eIdx] = 0.0;

            //numerator and denominator of L2 error
            numerator += volume*(approxPressure - exactPressure)*(approxPressure - exactPressure);
            denominator += volume*exactPressure*exactPressure;

            // update uMinExact and uMaxExact
            uMinExact = std::min(uMinExact, exactPressure);
            uMaxExact = std::max(uMaxExact, exactPressure);

            // update uMin and uMax
            uMin = std::min(uMin, approxPressure);
            uMax = std::max(uMax, approxPressure);

            // get the absolute permeability
            auto K = problem.spatialParams().permeability(element, scv,
                     problem.model().elementSolution(element, problem.model().curSol()));

            Dune::FieldVector<Scalar,dim> exactGradient(0.0);

            auto elemVolVars = localView(problem.model().curGlobalVolVars());
            elemVolVars.bind(element, fvGeometry, problem.model().curSol());

            auto prevElemVolVars = localView(problem.model().prevGlobalVolVars());
            prevElemVolVars.bind(element, fvGeometry, problem.model().prevSol());

            auto elemFluxVarsCache = localView(problem.model().globalFluxVarsCache());
            elemFluxVarsCache.bindElement(element, fvGeometry, elemVolVars);

            auto prevElemFluxVarsCache = localView(problem.model().globalFluxVarsCache());
            prevElemFluxVarsCache.bindElement(element, fvGeometry, prevElemVolVars);

            auto& curGlobalFaceVars = problem.model().curGlobalFaceVars();
            auto& prevGlobalFaceVars = problem.model().prevGlobalFaceVars();

            //loops over faces of element
            for (auto&& scvf : scvfs(fvGeometry))
            {
                Scalar faceVol = scvf.area();
                const auto insideScvIdx = scvf.insideScvIdx();
                const auto& insideScv = fvGeometry.scv(insideScvIdx);

                auto unitOuterNormal = scvf.unitOuterNormal();
                auto faceCenter = scvf.center();

                // get the exact gradient at the cell center: grad p_ex
                exactGradient = problem.exactGrad(faceCenter, cellCenter);

                Dune::FieldVector<Scalar,dim> KGrad(0);
                K.mv(exactGradient, KGrad);

                // calculate the exact normal velocity v = -K grad p * n
                Scalar exactVel = KGrad*unitOuterNormal;
                exactVel *= -1;

                const auto& insideVolVars = elemVolVars[insideScvIdx];

                FluxVariables fluxVars;
                fluxVars.init(problem, element, fvGeometry, elemVolVars, curGlobalFaceVars, scvf, elemFluxVarsCache);

                // Assuming that the initial Data is the exact solution
                FluxVariables fluxVarsExact;
                fluxVarsExact.init(problem, element, fvGeometry, prevElemVolVars, prevGlobalFaceVars, scvf, prevElemFluxVarsCache);


                // the physical quantities for which we perform upwinding
                auto upwindTerm = [](const auto& volVars)
                                  { return volVars.density(0)*volVars.mobility(0); };

                PrimaryVariables flux(fluxVars.advectiveFlux(0, upwindTerm));

                PrimaryVariables fluxExact(fluxVarsExact.advectiveFlux(0, upwindTerm));

                Scalar approximateVel = flux[conti0EqIdx]/faceVol;

                // calculate the difference in the normal velocity
                Scalar velDiff = exactVel - approximateVel;

                // calculate the fluxes through the element faces
                Scalar exactFlux = faceVol*exactVel;
                Scalar approximateFlux = faceVol*approximateVel;
                Scalar fluxDiff = faceVol*velDiff;

                Scalar faceDist = std::abs(unitOuterNormal*(faceCenter - cellCenter));

                if(scvf.boundary()){
                    Scalar facePressure = problem.dirichletAtPos(faceCenter)[cellCenterIdx];

                    Scalar valDiffApprox = (facePressure-approxPressure);
                    Scalar valDiffExact = (facePressure-exactPressure);
                    absH1ErrorApproxMin += faceVol/faceDist*valDiffApprox*valDiffApprox;
                    absH1ErrorDiffMin += faceVol/faceDist*(valDiffApprox-valDiffExact)*(valDiffApprox-valDiffExact);

                    //Different signs are used than in the paper
                    bilinearFormApprox += flux[conti0EqIdx]*approxPressure;
                    bilinearFormDiffApprox += (flux[conti0EqIdx]-fluxExact[conti0EqIdx])*(approxPressure-exactPressure);

                    Scalar associatedVolume = faceDist * faceVol;
                    numeratorFaceFlux += fluxDiff*fluxDiff*associatedVolume;
                    denominatorFaceFlux += exactFlux*exactFlux*associatedVolume;

                    numeratorFaceFluxBound +=  fluxDiff*fluxDiff*associatedVolume;
                    denominatorFaceFluxBound += exactFlux*exactFlux*associatedVolume;

                    denominatorFaceFlux_relVol += associatedVolume;

                    numeratorFaceVel += associatedVolume*velDiff*velDiff;
                    totalVelSum += associatedVolume*exactVel*exactVel;

                    localRelativeFaceFluxDiff[eIdx] += fluxDiff*fluxDiff*associatedVolume;
                    localRelativeFaceVelError[eIdx] += associatedVolume*velDiff*velDiff;

                    absL2ErrorFaceFluxes += fluxDiff*fluxDiff;

                }else{
                    const auto outsideScvIdx = scvf.outsideScvIdx();
                    const auto& outsideScv = fvGeometry.scv(outsideScvIdx);
                    const auto& outsideVolVars = elemVolVars[outsideScvIdx];

                    const auto outsideElement = fvGeometry.globalFvGeometry().element(outsideScvIdx);
                    int eIdxJ = problem.elementMapper().index(outsideElement);

                    Scalar exactPressureJ = problem.exact(outsideScv.center());
                    Scalar approxPressureJ = problem.model().curSol()[cellCenterIdx][outsideScv.elementIndex()][pressureIdx];
                    Scalar faceDistL = std::abs(unitOuterNormal*(faceCenter - outsideScv.center()));
                    Scalar valMinDiffApprox = faceDist/(faceDist + faceDistL)*(approxPressureJ-approxPressure);
                    Scalar valMinDiffExact = faceDist/(faceDist + faceDistL)*(exactPressureJ-exactPressure);

                    absH1ErrorApproxMin += faceVol/faceDist*valMinDiffApprox*valMinDiffApprox;
                    absH1ErrorDiffMin += faceVol/faceDist*(valMinDiffApprox-valMinDiffExact)*(valMinDiffApprox-valMinDiffExact);

                    //Different signs are used than in the paper
                    bilinearFormApprox += flux[conti0EqIdx]*approxPressure;
                    bilinearFormDiffApprox += (flux[conti0EqIdx]-fluxExact[conti0EqIdx])*(approxPressure-exactPressure);

                    if(!alreadyVisited[eIdxJ]){
                        absL2ErrorFaceFluxes += fluxDiff*fluxDiff;
                        //get volume of neighbored element
                        Scalar associatedVolumeJ = faceDistL * faceVol;
                        Scalar associatedVolumeI = faceDist * faceVol;
                        Scalar volumeAverage = 0.5*(associatedVolumeJ + associatedVolumeI);

                        numeratorFaceFlux +=  fluxDiff*fluxDiff*volumeAverage;
                        denominatorFaceFlux += exactFlux*exactFlux*volumeAverage;

                        numeratorFaceFluxInner +=  fluxDiff*fluxDiff*volumeAverage;
                        denominatorFaceFluxInner += exactFlux*exactFlux*volumeAverage;

                        numeratorFaceVel += volumeAverage*velDiff*velDiff;
                        totalVelSum += volumeAverage*exactVel*exactVel;

                        localRelativeFaceFluxDiff[eIdx] += fluxDiff*fluxDiff*volumeAverage;
                        localRelativeFaceVelError[eIdx] += volumeAverage*velDiff*velDiff;

                        localRelativeFaceFluxDiff[eIdxJ] += fluxDiff*fluxDiff*volumeAverage;
                        localRelativeFaceVelError[eIdxJ] += volumeAverage*velDiff*velDiff;

                        denominatorFaceFlux_relVol += volumeAverage;

                    }
                }
            }

            // calculate the maximum of the diagonal length of all elements on leaf grid
            for (int i = 0; i < element.geometry().corners(); ++i)
            {
                auto corner1 = element.geometry().corner(i);

                for (int j = 0; j < element.geometry().corners(); ++j)
                {
                    // get all corners of current element and compare the distances between them
                    auto corner2 = element.geometry().corner(j);

                    // distance vector between corners
                    auto distVec = corner1 - corner2;
                    Scalar dist = distVec.two_norm();

                    if (hMax < dist)
                        hMax = dist;
                    if(i!=j && dist > 1.0e-30)
                        hMin = std::min(hMin,dist);
                }
            }

            alreadyVisited[eIdx] = true;
        }

        absH1ErrorApproxMin = std::sqrt(absH1ErrorApproxMin);
        absH1ErrorDiffMin = std::sqrt(absH1ErrorDiffMin);
        relativeL2Error = std::sqrt(numerator/denominator);
        absL2Error = std::sqrt(numerator);

        relativeL2ErrorFaceFluxes = std::sqrt(numeratorFaceFlux/denominatorFaceFlux);
        relVelL2Error = std::sqrt(numeratorFaceVel/totalVelSum);
        relVolVelL2Error = std::sqrt(numeratorFaceVel/denominatorFaceFlux_relVol);
        absL2ErrorFaceFluxes = std::sqrt(absL2ErrorFaceFluxes);

        localRelativeFaceFluxDiff /= denominatorFaceFlux;
        localRelativeFaceVelError /= totalVelSum;

        relativeL2ErrorFaceFluxesBound = std::sqrt(numeratorFaceFluxBound/denominatorFaceFluxBound);
        relativeL2ErrorFaceFluxesInner = std::sqrt(numeratorFaceFluxInner/denominatorFaceFluxInner);

        return;
    }

};

}

#endif
