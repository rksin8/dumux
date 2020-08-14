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
 * \ingroup BoxDiscretization
 * \ingroup MultiDomain
 * \brief An assembler for Jacobian and residual contribution per element (box methods) for multidomain problems
 */
#ifndef DUMUX_MULTIDOMAIN_FACECENTERED_LOCAL_ASSEMBLER_HH
#define DUMUX_MULTIDOMAIN_FACECENTERED_LOCAL_ASSEMBLER_HH

#include <dune/common/reservedvector.hh>
#include <dune/common/indices.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/grid/common/gridenums.hh> // for GhostEntity
#include <dune/istl/matrixindexset.hh>

#include <dumux/common/reservedblockvector.hh>
#include <dumux/common/properties.hh>
#include <dumux/common/parameters.hh>
#include <dumux/common/numericdifferentiation.hh>
#include <dumux/assembly/numericepsilon.hh>
#include <dumux/assembly/diffmethod.hh>
#include <dumux/assembly/fvlocalassemblerbase.hh>

namespace Dumux {

/*!
 * \ingroup Assembly
 * \ingroup BoxDiscretization
 * \ingroup MultiDomain
 * \brief A base class for all box local assemblers
 * \tparam id the id of the sub domain
 * \tparam TypeTag the TypeTag
 * \tparam Assembler the assembler type
 * \tparam Implementation the actual implementation type
 * \tparam implicit Specifies whether the time discretization is implicit or not not (i.e. explicit)
 */
template<std::size_t id, class TypeTag, class Assembler, class Implementation, bool implicit>
class SubDomainFaceCenteredLocalAssemblerBase : public FVLocalAssemblerBase<TypeTag, Assembler, Implementation, implicit>
{
    using ParentType = FVLocalAssemblerBase<TypeTag, Assembler,Implementation, implicit>;

    using Problem = GetPropType<TypeTag, Properties::Problem>;
    using LocalResidualValues = GetPropType<TypeTag, Properties::NumEqVector>;
    using JacobianMatrix = GetPropType<TypeTag, Properties::JacobianMatrix>;
    using SolutionVector = typename Assembler::SolutionVector;
    using SubSolutionVector = GetPropType<TypeTag, Properties::SolutionVector>;
    using ElementBoundaryTypes = GetPropType<TypeTag, Properties::ElementBoundaryTypes>;

    using GridVariables = GetPropType<TypeTag, Properties::GridVariables>;
    using GridVolumeVariables = typename GridVariables::GridVolumeVariables;
    using ElementVolumeVariables = typename GridVolumeVariables::LocalView;
    using ElementFluxVariablesCache = typename GridVariables::GridFluxVariablesCache::LocalView;
    using Scalar = typename GridVariables::Scalar;

    using GridGeometry = typename GridVariables::GridGeometry;
    using FVElementGeometry = typename GridGeometry::LocalView;
    using SubControlVolume = typename GridGeometry::SubControlVolume;
    using SubControlVolumeFace = typename GridGeometry::SubControlVolumeFace;
    using GridView = typename GridGeometry::GridView;
    using Element = typename GridView::template Codim<0>::Entity;

    using CouplingManager = typename Assembler::CouplingManager;

    static constexpr auto numEq = GetPropType<TypeTag, Properties::ModelTraits>::numEq();

public:
    //! export the domain id of this sub-domain
    static constexpr auto domainId = typename Dune::index_constant<id>();
    //! pull up constructor of parent class
    using ParentType::ParentType;
    //! export element residual vector type
    using ElementResidualVector = typename ParentType::LocalResidual::ElementResidualVector;

    // the constructor
    explicit SubDomainFaceCenteredLocalAssemblerBase(const Assembler& assembler,
                                                     const Element& element,
                                                     const SolutionVector& curSol,
                                                     CouplingManager& couplingManager)
    : ParentType(assembler,
                 element,
                 curSol,
                 localView(assembler.gridGeometry(domainId)),
                 localView(assembler.gridVariables(domainId).curGridVolVars()),
                 localView(assembler.gridVariables(domainId).prevGridVolVars()),
                 localView(assembler.gridVariables(domainId).gridFluxVarsCache()),
                 assembler.localResidual(domainId),
                 (element.partitionType() == Dune::GhostEntity))
    , couplingManager_(couplingManager)
    {}

    /*!
     * \brief Computes the derivatives with respect to the given element and adds them
     *        to the global matrix. The element residual is written into the right hand side.
     */
    template<class JacobianMatrixRow, class GridVariablesTuple>
    void assembleJacobianAndResidual(JacobianMatrixRow& jacRow, SubSolutionVector& res, GridVariablesTuple& gridVariables)
    {
        this->asImp_().bindLocalViews();
        this->elemBcTypes().update(problem(), this->element(), this->fvGeometry());

        // for the diagonal jacobian block
        // forward to the internal implementation
        const auto residual = this->asImp_().assembleJacobianAndResidualImpl(jacRow[domainId], *std::get<domainId>(gridVariables));

        // update the residual vector
        for (const auto& scv : scvs(this->fvGeometry()))
            res[scv.dofIndex()] += residual[scv.localDofIndex()];

        // assemble the coupling blocks
        using namespace Dune::Hybrid;
        forEach(integralRange(Dune::Hybrid::size(jacRow)), [&](auto&& i)
        {
            if (i != id)
                this->assembleJacobianCoupling(i, jacRow, residual, gridVariables);
        });

        // lambda for the incorporation of Dirichlet Bcs
        auto applyDirichlet = [&] (const auto& scvI,
                                   const auto& dirichletValues,
                                   const auto eqIdx,
                                   const auto pvIdx)
        {
            res[scvI.dofIndex()][eqIdx] = this->curElemVolVars()[scvI].priVars()[pvIdx] - dirichletValues[pvIdx];

            // in explicit schemes we only have entries on the diagonal
            // and thus don't have to do anything with off-diagonal entries
            if (implicit)
            {
                auto& row = jacRow[domainId][scvI.dofIndex()];
                for (auto col = row.begin(); col != row.end(); ++col)
                    row[col.index()][eqIdx] = 0.0;
            }

            jacRow[domainId][scvI.dofIndex()][scvI.dofIndex()][eqIdx][pvIdx] = 1.0;
        };

        // incorporate Dirichlet BCs
        this->asImp_().enforceDirichletConstraints(applyDirichlet);
    }

    /*!
     * \brief Assemble the entries in a coupling block of the jacobian.
     *        There is no coupling block between a domain and itself.
     */
    template<std::size_t otherId, class JacRow, class GridVariables,
             typename std::enable_if_t<(otherId == id), int> = 0>
    void assembleJacobianCoupling(Dune::index_constant<otherId> domainJ, JacRow& jacRow,
                                  const ElementResidualVector& res, GridVariables& gridVariables)
    {}

    /*!
     * \brief Assemble the entries in a coupling block of the jacobian.
     */
    template<std::size_t otherId, class JacRow, class GridVariables,
             typename std::enable_if_t<(otherId != id), int> = 0>
    void assembleJacobianCoupling(Dune::index_constant<otherId> domainJ, JacRow& jacRow,
                                  const ElementResidualVector& res, GridVariables& gridVariables)
    {
        this->asImp_().assembleJacobianCoupling(domainJ, jacRow[domainJ], res, *std::get<domainId>(gridVariables));
    }

    /*!
     * \brief Assemble the residual vector entries only
     */
    void assembleResidual(SubSolutionVector& res)
    {
        this->asImp_().bindLocalViews();
        this->elemBcTypes().update(problem(), this->element(), this->fvGeometry());

        const auto residual = this->evalLocalResidual();
        for (const auto& scv : scvs(this->fvGeometry()))
            res[scv.dofIndex()] += residual[scv.localDofIndex()];

        auto applyDirichlet = [&] (const auto& scvI,
                                   const auto& dirichletValues,
                                   const auto eqIdx,
                                   const auto pvIdx)
        {
            res[scvI.dofIndex()][eqIdx] = this->curElemVolVars()[scvI].priVars()[pvIdx] - dirichletValues[pvIdx];
        };

        this->asImp_().enforceDirichletConstraints(applyDirichlet);
    }

    /*!
     * \brief Evaluates the local source term for an element and given element volume variables
     */
    ElementResidualVector evalLocalSourceResidual(const Element& element, const ElementVolumeVariables& elemVolVars) const
    {
        // initialize the residual vector for all scvs in this element
        ElementResidualVector residual(this->fvGeometry().numScv());

        // evaluate the volume terms (storage + source terms)
        // forward to the local residual specialized for the discretization methods
        for (auto&& scv : scvs(this->fvGeometry()))
        {
            const auto& curVolVars = elemVolVars[scv];
            auto source = this->localResidual().computeSource(problem(), element, this->fvGeometry(), elemVolVars, scv);
            source *= -scv.volume()*curVolVars.extrusionFactor();
            residual[scv.localDofIndex()] = std::move(source);
        }

        return residual;
    }

    /*!
     * \brief Evaluates the local source term depending on time discretization scheme
     */
    ElementResidualVector evalLocalSourceResidual(const Element& neighbor) const
    { return this->evalLocalSourceResidual(neighbor, implicit ? this->curElemVolVars() : this->prevElemVolVars()); }

    //! Enforce Dirichlet constraints
    template<typename ApplyFunction>
    void enforceDirichletConstraints(const ApplyFunction& applyDirichlet)
    {
        // enforce Dirichlet boundary conditions
        this->asImp_().evalDirichletBoundaries(applyDirichlet);
        // take care of internal Dirichlet constraints (if enabled)
        // this->asImp_().enforceInternalDirichletConstraints(applyDirichlet);
    }

    /*!
     * \brief Incorporate Dirichlet boundary conditions
     * \param applyDirichlet Lambda function for the BC incorporation on an scv
     */
    template< typename ApplyDirichletFunctionType >
    void evalDirichletBoundaries(ApplyDirichletFunctionType applyDirichlet)
    {
        // enforce Dirichlet boundaries by overwriting partial derivatives with 1 or 0
        // and set the residual to (privar - dirichletvalue)
        if (this->elemBcTypes().hasDirichlet())
        {
            for (const auto& scvf : scvfs(this->fvGeometry()))
            {
                if (scvf.isFrontal() && scvf.boundary())
                {
                    const auto bcTypes = this->elemBcTypes()[scvf.localIndex()];
                    if (bcTypes.hasDirichlet())
                    {
                        const auto& scv = this->fvGeometry().scv(scvf.insideScvIdx());
                        using PrimaryVariables = GetPropType<TypeTag, Properties::PrimaryVariables>;
                        const PrimaryVariables dirichletValues(this->problem().dirichlet(this->element(), scvf)[scv.directionIndex()]); // TODO

                        // set the Dirichlet conditions in residual and jacobian
                        for (int eqIdx = 0; eqIdx < numEq; ++eqIdx)
                        {
                            if (bcTypes.isDirichlet(eqIdx))
                            {
                                const auto pvIdx = bcTypes.eqToDirichletIndex(eqIdx);
                                assert(0 <= pvIdx && pvIdx < numEq);
                                applyDirichlet(scv, dirichletValues, eqIdx, pvIdx);
                            }
                        }
                    }
                }
            }
        }
    }

    /*!
     * \brief Prepares all local views necessary for local assembly.
     */
    void bindLocalViews()
    {
        // get some references for convenience
        const auto& element = this->element();
        const auto& curSol = this->curSol()[domainId];
        auto&& fvGeometry = this->fvGeometry();
        auto&& curElemVolVars = this->curElemVolVars();
        auto&& elemFluxVarsCache = this->elemFluxVarsCache();

        // bind the caches
        couplingManager_.bindCouplingContext(domainId, element, this->assembler());
        fvGeometry.bind(element);

        if (implicit)
        {
            curElemVolVars.bind(element, fvGeometry, curSol);
            elemFluxVarsCache.bind(element, fvGeometry, curElemVolVars);
            if (!this->assembler().isStationaryProblem())
                this->prevElemVolVars().bindElement(element, fvGeometry, this->assembler().prevSol()[domainId]);
        }
        else
        {
            auto& prevElemVolVars = this->prevElemVolVars();
            const auto& prevSol = this->assembler().prevSol()[domainId];

            curElemVolVars.bindElement(element, fvGeometry, curSol);
            prevElemVolVars.bind(element, fvGeometry, prevSol);
            elemFluxVarsCache.bind(element, fvGeometry, prevElemVolVars);
        }
    }

    //! return reference to the underlying problem
    const Problem& problem() const
    { return this->assembler().problem(domainId); }

    //! return reference to the coupling manager
    CouplingManager& couplingManager()
    { return couplingManager_; }

private:
    CouplingManager& couplingManager_; //!< the coupling manager
};

/*!
 * \ingroup Assembly
 * \ingroup BoxDiscretization
 * \ingroup MultiDomain
 * \brief The box scheme multidomain local assembler
 * \tparam id the id of the sub domain
 * \tparam TypeTag the TypeTag
 * \tparam Assembler the assembler type
 * \tparam DM the numeric differentiation method
 * \tparam implicit whether the assembler is explicit or implicit in time
 */
template<std::size_t id, class TypeTag, class Assembler, DiffMethod DM = DiffMethod::numeric, bool implicit = true>
class SubDomainFaceCenteredLocalAssembler;

/*!
 * \ingroup Assembly
 * \ingroup BoxDiscretization
 * \ingroup MultiDomain
 * \brief Box scheme multi domain local assembler using numeric differentiation and implicit time discretization
 */
template<std::size_t id, class TypeTag, class Assembler>
class SubDomainFaceCenteredLocalAssembler<id, TypeTag, Assembler, DiffMethod::numeric, /*implicit=*/true>
: public SubDomainFaceCenteredLocalAssemblerBase<id, TypeTag, Assembler,
             SubDomainFaceCenteredLocalAssembler<id, TypeTag, Assembler, DiffMethod::numeric, true>, true >
{
    using ThisType = SubDomainFaceCenteredLocalAssembler<id, TypeTag, Assembler, DiffMethod::numeric, /*implicit=*/true>;
    using ParentType = SubDomainFaceCenteredLocalAssemblerBase<id, TypeTag, Assembler, ThisType, /*implicit=*/true>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using VolumeVariables = GetPropType<TypeTag, Properties::VolumeVariables>;

    using ElementVolumeVariables = typename GetPropType<TypeTag, Properties::GridVolumeVariables>::LocalView;
    using GridGeometry = GetPropType<TypeTag, Properties::GridGeometry>;
    using FVElementGeometry = typename GridGeometry::LocalView;
    using GridView = typename GridGeometry::GridView;
    using Element = typename GridView::template Codim<0>::Entity;
    using SubControlVolume = typename FVElementGeometry::SubControlVolume;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;

    enum { numEq = GetPropType<TypeTag, Properties::ModelTraits>::numEq() };
    enum { dim = GridView::dimension };

    static constexpr bool enableGridFluxVarsCache = getPropValue<TypeTag, Properties::EnableGridFluxVariablesCache>();
    static constexpr bool enableGridVolVarsCache = getPropValue<TypeTag, Properties::EnableGridVolumeVariablesCache>();
    static constexpr auto domainI = Dune::index_constant<id>();

public:
    using ParentType::ParentType;
    //! export element residual vector type
    using ElementResidualVector = typename ParentType::LocalResidual::ElementResidualVector;

    /*!
     * \brief Computes the derivatives with respect to the given element and adds them
     *        to the global matrix.
     *
     * \return The element residual at the current solution.
     */
    template<class JacobianMatrixDiagBlock, class GridVariables>
    ElementResidualVector assembleJacobianAndResidualImpl(JacobianMatrixDiagBlock& A, GridVariables& gridVariables)
    {
        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Calculate derivatives of all dofs in stencil with respect to the dofs in the element. In the //
        // neighboring elements we do so by computing the derivatives of the fluxes which depend on the //
        // actual element. In the actual element we evaluate the derivative of the entire residual.     //
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // get the vector of the actual element residuals
        const auto origResiduals = this->evalLocalResidual();

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // compute the derivatives of this element with respect to all of the element's dofs and add them to the Jacobian //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        const auto& problem = this->problem();
        const auto& element = this->element();
        const auto& fvGeometry = this->fvGeometry();
        const auto& curSol = this->curSol()[domainI];
        auto&& curElemVolVars = this->curElemVolVars();

        // create the element solution
        auto elemSol = elementSolution(element, curSol, fvGeometry.gridGeometry());

        auto partialDerivs = origResiduals;
        partialDerivs = 0.0;

        for (auto&& scv : scvs(fvGeometry))
        {
            // dof index and corresponding actual pri vars
            const auto dofIdx = scv.dofIndex();
            auto& curVolVars = this->getVolVarAccess(gridVariables.curGridVolVars(), curElemVolVars, scv);
            const VolumeVariables origVolVars(curVolVars);

            auto evalSource = [&](ElementResidualVector& residual, const SubControlVolume& scv)
            {
                this->localResidual().evalSource(residual, problem, element, fvGeometry, curElemVolVars, scv);
            };

            auto evalStorage = [&](ElementResidualVector& residual, const SubControlVolume& scv)
            {
                this->localResidual().evalStorage(residual, problem, element, fvGeometry, this->prevElemVolVars(), curElemVolVars, scv);
            };

            auto evalFlux = [&](ElementResidualVector& residual, const SubControlVolumeFace& scvf)
            {
                this->localResidual().evalFlux(residual, problem, element, fvGeometry, curElemVolVars, this->elemBcTypes(), this->elemFluxVarsCache(), scvf);
            };

            // derivative w.r.t. own DOF
            for (int pvIdx = 0; pvIdx < numEq; pvIdx++)
            {
                partialDerivs = 0.0;

                auto evalResiduals = [&](Scalar priVar)
                {
                    // update the volume variables and compute element residual
                    elemSol[scv.localDofIndex()][pvIdx] = priVar;
                    curVolVars.update(elemSol, problem, element, scv);
                    this->couplingManager().updateCouplingContext(domainI, *this, domainI, scv.dofIndex(), elemSol[scv.localDofIndex()], pvIdx);

                    ElementResidualVector residual(element.subEntities(1));
                    residual = 0;

                    evalSource(residual, scv);

                    if (!this->assembler().isStationaryProblem())
                        evalStorage(residual, scv);

                    for (const auto& scvf : scvfs(fvGeometry, scv))
                        evalFlux(residual, scvf);

                    return residual;
                };

                // derive the residuals numerically
                static const NumericEpsilon<Scalar, numEq> eps_{this->problem().paramGroup()};
                static const int numDiffMethod = getParamFromGroup<int>(this->problem().paramGroup(), "Assembly.NumericDifferenceMethod");
                NumericDifferentiation::partialDerivative(evalResiduals, elemSol[scv.localDofIndex()][pvIdx], partialDerivs, origResiduals,
                                                          eps_(elemSol[scv.localDofIndex()][pvIdx], pvIdx), numDiffMethod);

                for (int eqIdx = 0; eqIdx < numEq; eqIdx++)
                {
                    // A[i][col][eqIdx][pvIdx] is the rate of change of
                    // the residual of equation 'eqIdx' at dof 'i'
                    // depending on the primary variable 'pvIdx' at dof
                    // 'col'.
                    A[dofIdx][dofIdx][eqIdx][pvIdx] += partialDerivs[scv.localDofIndex()][eqIdx];
                }

                // restore the original state of the scv's volume variables
                curVolVars = origVolVars;

                // restore the original element solution
                elemSol[scv.localDofIndex()][pvIdx] = curSol[scv.dofIndex()][pvIdx];
                this->couplingManager().updateCouplingContext(domainI, *this, domainI, scv.dofIndex(), elemSol[scv.localDofIndex()], pvIdx);
            }

            // derivative w.r.t. other DOFs
            const auto& otherScvIndices = fvGeometry.gridGeometry().connectivityMap()[scv.index()];
            for (const auto globalJ : otherScvIndices)
            {
                ElementResidualVector partialDerivsFluxOnly(element.subEntities(2));
                partialDerivsFluxOnly = 0;

                const auto scvJ = fvGeometry.scv(globalJ);
                auto& curOtherVolVars = this->getVolVarAccess(gridVariables.curGridVolVars(), curElemVolVars, scvJ);
                const VolumeVariables origOtherVolVars(curOtherVolVars);

                const auto& otherElement = fvGeometry.gridGeometry().element(scvJ.elementIndex());
                auto otherElemSol = elementSolution(otherElement, curSol, fvGeometry.gridGeometry()); // TODO allow selective creation of elemsol (for one scv)

                for (int pvIdx = 0; pvIdx < numEq; pvIdx++)
                {
                    partialDerivsFluxOnly = 0.0;

                    auto evalResiduals = [&](Scalar priVar)
                    {
                        // update the volume variables and compute element residual
                        otherElemSol[scvJ.localDofIndex()][pvIdx] = priVar;
                        curOtherVolVars.update(otherElemSol, problem, otherElement, scvJ);
                        this->couplingManager().updateCouplingContext(domainI, *this, domainI, scvJ.dofIndex(), otherElemSol[scvJ.localDofIndex()], pvIdx);


                        ElementResidualVector residual(element.subEntities(1));
                        residual = 0;

                        for (const auto& scvf : scvfs(fvGeometry, scv))
                        {
                            if (scvf.outsideScvIdx() == scvJ.index())
                            {
                                evalFlux(residual, scvf);
                                return residual;
                            }

                            if (scvf.isLateral())
                            {
                                const auto& orthogonalScvf = fvGeometry.lateralOrthogonalScvf(scvf);
                                if (orthogonalScvf.insideScvIdx() == scvJ.index() || orthogonalScvf.outsideScvIdx() == scvJ.index())
                                {
                                    evalFlux(residual, scvf);
                                    return residual;
                                }
                            }
                        }

                        DUNE_THROW(Dune::InvalidStateException, "No scvf found");
                    };

                    // get original fluxes
                    const auto origFluxResidual = [&]()
                    {
                        ElementResidualVector result(element.subEntities(2));
                        result = 0;

                        for (const auto& scvf : scvfs(fvGeometry, scv))
                        {
                            if (scvf.outsideScvIdx() == scvJ.index())
                            {
                                evalFlux(result, scvf);
                                return result;
                            }

                            if (scvf.isLateral())
                            {
                                const auto& orthogonalScvf = fvGeometry.lateralOrthogonalScvf(scvf);
                                if (orthogonalScvf.insideScvIdx() == scvJ.index() || orthogonalScvf.outsideScvIdx() == scvJ.index())
                                {
                                    evalFlux(result, scvf);
                                    return result;
                                }
                            }
                        }
                        DUNE_THROW(Dune::InvalidStateException, "No scvf found");
                    }();

                    // derive the residuals numerically
                    static const NumericEpsilon<Scalar, numEq> eps_{this->problem().paramGroup()};
                    static const int numDiffMethod = getParamFromGroup<int>(this->problem().paramGroup(), "Assembly.NumericDifferenceMethod");
                    NumericDifferentiation::partialDerivative(evalResiduals, otherElemSol[scvJ.localDofIndex()][pvIdx], partialDerivsFluxOnly, origFluxResidual,
                                                            eps_(otherElemSol[scvJ.localDofIndex()][pvIdx], pvIdx), numDiffMethod);

                    for (int eqIdx = 0; eqIdx < numEq; eqIdx++)
                    {
                        // A[i][col][eqIdx][pvIdx] is the rate of change of
                        // the residual of equation 'eqIdx' at dof 'i'
                        // depending on the primary variable 'pvIdx' at dof
                        // 'col'.
                        A[dofIdx][scvJ.dofIndex()][eqIdx][pvIdx] += partialDerivsFluxOnly[scv.localDofIndex()][eqIdx];
                    }

                    // restore the original state of the scv's volume variables
                    curOtherVolVars = origOtherVolVars;

                    // restore the original element solution
                    otherElemSol[scvJ.localDofIndex()][pvIdx] = curSol[scvJ.dofIndex()][pvIdx];
                    this->couplingManager().updateCouplingContext(domainI, *this, domainI, scvJ.dofIndex(), otherElemSol[scvJ.localDofIndex()], pvIdx);
                }
            }
        }

        // evaluate additional derivatives that might arise from the coupling
        this->couplingManager().evalAdditionalDomainDerivatives(domainI, *this, origResiduals, A, gridVariables);

        return origResiduals;
    }

    /*!
     * \brief Computes the derivatives with respect to the given element and adds them
     *        to the global matrix.
     *
     * \return The element residual at the current solution.
     */
    template<std::size_t otherId, class JacobianBlock, class GridVariables>
    void assembleJacobianCoupling(Dune::index_constant<otherId> domainJ, JacobianBlock& A,
                                  const ElementResidualVector& res, GridVariables& gridVariables)
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Calculate derivatives of all dofs in the element with respect to all dofs in the coupling stencil. //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        // get some aliases for convenience
        const auto& element = this->element();
        const auto& fvGeometry = this->fvGeometry();
        auto&& curElemVolVars = this->curElemVolVars();
        auto&& elemFluxVarsCache = this->elemFluxVarsCache();

        // the solution vector of the other domain
        const auto& curSolJ = this->curSol()[domainJ];

        // convenience lambda for call to update self
        auto updateCoupledVariables = [&] ()
        {
            // Update ourself after the context has been modified. Depending on the
            // type of caching, other objects might have to be updated. All ifs can be optimized away.
            if constexpr (enableGridFluxVarsCache)
            {
                if constexpr (enableGridVolVarsCache)
                    this->couplingManager().updateCoupledVariables(domainI, *this, gridVariables.curGridVolVars(), gridVariables.gridFluxVarsCache());
                else
                    this->couplingManager().updateCoupledVariables(domainI, *this, curElemVolVars, gridVariables.gridFluxVarsCache());
            }
            else
            {
                if constexpr (enableGridVolVarsCache)
                    this->couplingManager().updateCoupledVariables(domainI, *this, gridVariables.curGridVolVars(), elemFluxVarsCache);
                else
                    this->couplingManager().updateCoupledVariables(domainI, *this, curElemVolVars, elemFluxVarsCache);
            }
        };

        for (const auto& scv : scvs(fvGeometry))
        {
            const auto& stencil = this->couplingManager().couplingStencil(domainI, element, scv, domainJ);

            for (const auto globalJ : stencil)
            {
                const auto origResidual = this->couplingManager().evalCouplingResidual(domainI, *this, scv, domainJ, globalJ); // TODO is this necessary?
                // undeflected privars and privars to be deflected
                const auto origPriVarsJ = curSolJ[globalJ];
                auto priVarsJ = origPriVarsJ;

                for (int pvIdx = 0; pvIdx < JacobianBlock::block_type::cols; ++pvIdx)
                {
                    auto evalCouplingResidual = [&](Scalar priVar)
                    {
                        priVarsJ[pvIdx] = priVar;
                        this->couplingManager().updateCouplingContext(domainI, *this, domainJ, globalJ, priVarsJ, pvIdx);
                        updateCoupledVariables();
                        return this->couplingManager().evalCouplingResidual(domainI, *this, scv, domainJ, globalJ);
                    };

                    // derive the residuals numerically
                    ElementResidualVector partialDerivs(element.subEntities(1));

                    const auto& paramGroup = this->assembler().problem(domainJ).paramGroup();
                    static const int numDiffMethod = getParamFromGroup<int>(paramGroup, "Assembly.NumericDifferenceMethod");
                    static const auto epsCoupl = this->couplingManager().numericEpsilon(domainJ, paramGroup);

                    NumericDifferentiation::partialDerivative(evalCouplingResidual, origPriVarsJ[pvIdx], partialDerivs, origResidual,
                                                              epsCoupl(origPriVarsJ[pvIdx], pvIdx), numDiffMethod);

                    // update the global stiffness matrix with the current partial derivatives
                    for (int eqIdx = 0; eqIdx < numEq; eqIdx++)
                    {
                        // A[i][col][eqIdx][pvIdx] is the rate of change of
                        // the residual of equation 'eqIdx' at dof 'i'
                        // depending on the primary variable 'pvIdx' at dof
                        // 'col'.
                        A[scv.dofIndex()][globalJ][eqIdx][pvIdx] += partialDerivs[scv.localDofIndex()][eqIdx];
                    }

                    // handle Dirichlet boundary conditions
                    // TODO internal constraints
                    if (scv.boundary() && this->elemBcTypes().hasDirichlet())
                    {
                        const auto bcTypes = this->elemBcTypes()[fvGeometry.frontalScvfOnBoundary(scv).localIndex()];
                        if (bcTypes.hasDirichlet())
                        {
                            // If the dof is coupled by a Dirichlet condition,
                            // set the derived value only once (i.e. overwrite existing values).
                            // For other dofs, add the contribution of the partial derivative.
                            for (int eqIdx = 0; eqIdx < numEq; ++eqIdx)
                            {
                                if (bcTypes.isCouplingDirichlet(eqIdx))
                                    A[scv.dofIndex()][globalJ][eqIdx][pvIdx] = partialDerivs[scv.localDofIndex()][eqIdx];
                                else if (bcTypes.isDirichlet(eqIdx))
                                    A[scv.dofIndex()][globalJ][eqIdx][pvIdx] = 0.0;
                            }
                        }
                    }

                    // restore the current element solution
                    priVarsJ[pvIdx] = origPriVarsJ[pvIdx];

                    // restore the undeflected state of the coupling context
                    this->couplingManager().updateCouplingContext(domainI, *this, domainJ, globalJ, priVarsJ, pvIdx);
                }

                // Restore original state of the flux vars cache and/or vol vars.
                // This has to be done in case they depend on variables of domainJ before
                // we continue with the numeric derivative w.r.t the next globalJ. Otherwise,
                // the next "origResidual" will be incorrect.
                updateCoupledVariables();
            }
        }
    }
};

} // end namespace Dumux

#endif