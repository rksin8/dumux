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
 * \ingroup BoundaryTests
 * \brief A simple Darcy test problem (cell-centered finite volume method).
 */

#ifndef DUMUX_DARCY_SUBPROBLEM_HH
#define DUMUX_DARCY_SUBPROBLEM_HH

#include <dune/grid/yaspgrid.hh>

#include <dumux/discretization/cctpfa.hh>

#include <dumux/porousmediumflow/1p/model.hh>
#include <dumux/porousmediumflow/problem.hh>

#include "../spatialparams.hh"

#include <dumux/material/components/constant.hh>
#include <dumux/material/fluidsystems/1pliquid.hh>

namespace Dumux {
template <class TypeTag>
class DarcySubProblem;

namespace Properties {
// Create new type tags
namespace TTag {
struct DarcyOneP { using InheritsFrom = std::tuple<OneP, CCTpfaModel>; };
} // end namespace TTag

// Set the problem property
template<class TypeTag>
struct Problem<TypeTag, TTag::DarcyOneP> { using type = Dumux::DarcySubProblem<TypeTag>; };

// the fluid system
template<class TypeTag>
struct FluidSystem<TypeTag, TTag::DarcyOneP>
{
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using type = FluidSystems::OnePLiquid<Scalar, Dumux::Components::Constant<1, Scalar> > ;
};

// Set the grid type
template<class TypeTag>
struct Grid<TypeTag, TTag::DarcyOneP> { using type = Dune::YaspGrid<2>; };

template<class TypeTag>
struct SpatialParams<TypeTag, TTag::DarcyOneP>
{
    using FVGridGeometry = GetPropType<TypeTag, Properties::FVGridGeometry>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using type = OnePSpatialParams<FVGridGeometry, Scalar>;
};
} // end namespace Properties

template <class TypeTag>
class DarcySubProblem : public PorousMediumFlowProblem<TypeTag>
{
    using ParentType = PorousMediumFlowProblem<TypeTag>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using PrimaryVariables = GetPropType<TypeTag, Properties::PrimaryVariables>;
    using NumEqVector = GetPropType<TypeTag, Properties::NumEqVector>;
    using BoundaryTypes = GetPropType<TypeTag, Properties::BoundaryTypes>;
    using VolumeVariables = GetPropType<TypeTag, Properties::VolumeVariables>;
    using FVElementGeometry = typename GetPropType<TypeTag, Properties::FVGridGeometry>::LocalView;
    using SubControlVolume = typename FVElementGeometry::SubControlVolume;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;
    using FVGridGeometry = GetPropType<TypeTag, Properties::FVGridGeometry>;

    using Element = typename GridView::template Codim<0>::Entity;
    using GlobalPosition = typename Element::Geometry::GlobalCoordinate;

    using CouplingManager = GetPropType<TypeTag, Properties::CouplingManager>;

public:
    //! export the Indices
    using Indices = typename GetPropType<TypeTag, Properties::ModelTraits>::Indices;

    DarcySubProblem(std::shared_ptr<const FVGridGeometry> fvGridGeometry,
                   std::shared_ptr<CouplingManager> couplingManager)
    : ParentType(fvGridGeometry, "Darcy"), eps_(1e-7), couplingManager_(couplingManager)
    {
        problemName_  =  getParam<std::string>("Vtk.OutputName") + "_" + getParamFromGroup<std::string>(this->paramGroup(), "Problem.Name");
    }

    /*!
     * \brief The problem name.
     */
    const std::string& name() const
    {
        return problemName_;
    }

    /*!
     * \name Problem parameters
     */
    // \{

    /*!
     * \brief Returns the temperature within the domain in [K].
     *
     */
    Scalar temperature() const
    { return 273.15 + 10; } // 10°C
    // \}

    /*!
     * \name Boundary conditions
     */
    // \{

    /*!
      * \brief Specifies which kind of boundary condition should be
      *        used for which equation on a given boundary control volume.
      *
      * \param element The element
      * \param scvf The boundary sub control volume face
      */
    BoundaryTypes boundaryTypes(const Element &element, const SubControlVolumeFace &scvf) const
    {
        BoundaryTypes values;

        if (couplingManager().isCoupledEntity(CouplingManager::darcyIdx, scvf))
            values.setAllCouplingDirichlet();
        else
            values.setAllDirichlet();

        return values;
    }

        /*!
     * \brief Evaluates the boundary conditions for a Dirichlet control volume.
     *
     * \param element The element for which the Dirichlet boundary condition is set
     * \param scvf The boundary subcontrolvolumeface
     *
     * For this method, the \a values parameter stores primary variables.
     */
    PrimaryVariables dirichlet(const Element &element, const SubControlVolumeFace &scvf) const
    {
        if (couplingManager().isCoupledEntity(CouplingManager::darcyIdx, scvf))
            return couplingManager().couplingData().freeFlowInterfacePressure(element, scvf);
        else
            return PrimaryVariables(exactPressure(scvf.center()));
    }

    /*!
     * \brief Evaluates the boundary conditions for a Neumann control volume.
     *
     * \param element The element for which the Neumann boundary condition is set
     * \param fvGeometry The fvGeometry
     * \param elemVolVars The element volume variables
     * \param elemFluxVarsCache Flux variables caches for all faces in stencil
     * \param scvf The boundary sub control volume face
     *
     * For this method, the \a values variable stores primary variables.
     */
    template<class ElementVolumeVariables, class ElementFluxVarsCache>
    NumEqVector neumann(const Element& element,
                        const FVElementGeometry& fvGeometry,
                        const ElementVolumeVariables& elemVolVars,
                        const ElementFluxVarsCache& elemFluxVarsCache,
                        const SubControlVolumeFace& scvf) const
    {
        NumEqVector values(0.0);

        if (couplingManager().isCoupledEntity(CouplingManager::darcyIdx, scvf))
            values[Indices::conti0EqIdx] = couplingManager().couplingData().massCouplingCondition(element, fvGeometry, elemVolVars, scvf);

        return values;
    }

    // \}

    /*!
     * \name Volume terms
     */
    // \{
    /*!
     * \brief Evaluates the source term for all phases within a given
     *        sub control volume.
     *
     * \param element The element for which the source term is set
     * \param fvGeometry The fvGeometry
     * \param elemVolVars The element volume variables
     * \param scv The sub control volume
     */
    template<class ElementVolumeVariables>
    NumEqVector source(const Element &element,
                       const FVElementGeometry& fvGeometry,
                       const ElementVolumeVariables& elemVolVars,
                       const SubControlVolume &scv) const
    { return NumEqVector(0.0); }

    // \}

    /*!
     * \brief Evaluates the initial value for a control volume.
     *
     * \param element The element
     *
     * For this method, the \a priVars parameter stores primary
     * variables.
     */
    PrimaryVariables initial(const Element &element) const
    {
        return PrimaryVariables(0.0);
    }

    //! exact pressure solution at a given position
    Scalar exactPressure(const GlobalPosition& globalPos) const
    {
        const Scalar x = globalPos[0];
        const Scalar y = globalPos[1];
        return x*(1.0-x)*(y-1.0) + (y-1.0)*(y-1.0)*(y-1.0)/3.0 + 2.0*x + 2.0*y + 4.0;
    }

    GlobalPosition exactVelocity(const GlobalPosition& globalPos) const
    {
        const Scalar x = globalPos[0];
        const Scalar y = globalPos[1];
        GlobalPosition velocity(0.0);
        velocity[0] = x*(y - 1.0) + (x - 1.0)*(y - 1.0) - 2.0;
        velocity[1] = x*(x - 1.0) - 1.0*(y - 1.0)*(y - 1.0) - 2.0;
        return velocity;
    }


    /*!
     * \brief Returns the analytical solution of the problem at a given position.
     *
     * \param globalPos The global position
     * \param time The current simulation time
     */
    auto analyticalSolution(const GlobalPosition& globalPos) const
    {
        // see Shiue et al., 2018: "Convergence of the MAC Scheme for the Stokes/Darcy Coupling Problem"
        Dune::FieldVector<Scalar, 3> values;
        const Scalar p = exactPressure(globalPos);
        const auto v = exactVelocity(globalPos);

        values[2] = p;
        values[0] = v[0];
        values[1] = v[1];

        return values;
    }


    // \}

    //! Get the coupling manager
    const CouplingManager& couplingManager() const
    { return *couplingManager_; }

private:

    bool onLowerBoundary_(const GlobalPosition &globalPos) const
    { return globalPos[1] < this->fvGridGeometry().bBoxMin()[1] + eps_; }

    Scalar eps_;
    std::shared_ptr<CouplingManager> couplingManager_;
    std::string problemName_;
    bool verticalFlow_;
};
} // end namespace Dumux

#endif //DUMUX_DARCY_SUBPROBLEM_HH