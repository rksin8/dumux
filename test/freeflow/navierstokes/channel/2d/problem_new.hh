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
 * \ingroup NavierStokesTests
 * \brief Channel flow test for the staggered grid (Navier-)Stokes model.
 */

#ifndef DUMUX_CHANNEL_TEST_PROBLEM_HH
#define DUMUX_CHANNEL_TEST_PROBLEM_HH

#include <dune/grid/yaspgrid.hh>



#include <dumux/material/fluidsystems/1pgas.hh>
#include <dumux/material/components/constant.hh>
#include <dumux/material/components/air.hh>

#include <dumux/freeflow/navierstokes/momentum/model.hh>
#include <dumux/freeflow/navierstokes/mass/1p/model.hh>
#include <dumux/freeflow/navierstokes/problem.hh>
#include <dumux/discretization/fcstaggered.hh>
#include <dumux/discretization/cctpfa.hh>

namespace Dumux {
template <class TypeTag>
class ChannelTestProblem;

namespace Properties {
// // Create new type tags
// namespace TTag {
// #if !NONISOTHERMAL
// struct ChannelTest { using InheritsFrom = std::tuple<NavierStokes, StaggeredFreeFlowModel>; };
// #else
// struct ChannelTest { using InheritsFrom = std::tuple<NavierStokesNI, StaggeredFreeFlowModel>; };
// #endif
// } // end namespace TTag
// Create new type tags
namespace TTag {
struct ChannelTest {};
struct ChannelTestMomentum { using InheritsFrom = std::tuple<ChannelTest, NavierStokesMomentum, FaceCenteredStaggeredModel>; };
struct ChannelTestMass { using InheritsFrom = std::tuple<ChannelTest, NavierStokesMassOneP, CCTpfaModel>; };
} // end namespace TTag

// Set the problem property
template<class TypeTag>
struct Problem<TypeTag, TTag::ChannelTest>
{
    using type = Dumux::ChannelTestProblem<TypeTag> ;
};

// the fluid system
template<class TypeTag>
struct FluidSystem<TypeTag, TTag::ChannelTest>
{
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
#if NONISOTHERMAL
    using type = FluidSystems::OnePLiquid<Scalar, Components::SimpleH2O<Scalar> >;
#else
    using type = FluidSystems::OnePGas<Scalar, Components::Air<Scalar> >;
#endif
};

// Set the grid type
template<class TypeTag>
struct Grid<TypeTag, TTag::ChannelTest> { using type = Dune::YaspGrid<2>; };


template<class TypeTag>
struct EnableGridGeometryCache<TypeTag, TTag::ChannelTest> { static constexpr bool value = true; };

template<class TypeTag>
struct EnableGridFluxVariablesCache<TypeTag, TTag::ChannelTest> { static constexpr bool value = true; };
template<class TypeTag>
struct EnableGridVolumeVariablesCache<TypeTag, TTag::ChannelTest> { static constexpr bool value = true; };
} // end namespace Properties


/*!
 * \ingroup NavierStokesTests
 * \brief  Test problem for the one-phase (Navier-) Stokes problem in a channel.
 *
 * Flow from left to right in a two-dimensional channel is considered. At the inlet (left),
 * fixed values for velocity are set, while at the outlet (right), a fixed pressure
 * boundary condition is used. The channel is confined by solid walls at the top and bottom
 * of the domain which corresponds to no-slip/no-flow conditions.
 * For the non-isothermal test, water of increased temperature is injected at the inlet
 * while the walls are fully isolating.
 */
template <class TypeTag>
class ChannelTestProblem : public NavierStokesProblem<TypeTag>
{
    using ParentType = NavierStokesProblem<TypeTag>;
    using BoundaryTypes = typename ParentType::BoundaryTypes;
    using GridGeometry = GetPropType<TypeTag, Properties::GridGeometry>;
    using FVElementGeometry = typename GridGeometry::LocalView;
    using SubControlVolume = typename FVElementGeometry::SubControlVolume;
    using SubControlVolumeFace = typename FVElementGeometry::SubControlVolumeFace;
    using Indices = typename GetPropType<TypeTag, Properties::ModelTraits>::Indices;
    using NumEqVector = typename ParentType::NumEqVector;
    using ModelTraits = GetPropType<TypeTag, Properties::ModelTraits>;
    using PrimaryVariables = typename ParentType::PrimaryVariables;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using SolutionVector = GetPropType<TypeTag, Properties::SolutionVector>;

    static constexpr auto dimWorld = GridGeometry::GridView::dimensionworld;
    using Element = typename GridGeometry::GridView::template Codim<0>::Entity;
    using GlobalPosition = typename Element::Geometry::GlobalCoordinate;
    using VelocityVector = Dune::FieldVector<Scalar, dimWorld>;

    using CouplingManager = GetPropType<TypeTag, Properties::CouplingManager>;
    using TimeLoopPtr = std::shared_ptr<CheckPointTimeLoop<Scalar>>;

    // the types of outlet boundary conditions
    enum class OutletCondition
    {
        outflow, doNothing, neumannXdirichletY, neumannXneumannY
    };

public:
    ChannelTestProblem(std::shared_ptr<const GridGeometry> gridGeometry, std::shared_ptr<CouplingManager> couplingManager)
    : ParentType(gridGeometry, couplingManager)
    , couplingManager_(couplingManager)
    {
        inletVelocity_ = getParam<Scalar>("Problem.InletVelocity");
        const auto tmp = getParam<std::string>("Problem.OutletCondition", "Outflow");
        if (tmp == "Outflow")
            outletCondition_ = OutletCondition::outflow;
        else if (tmp == "DoNothing")
            outletCondition_ = OutletCondition::doNothing;
        else if (tmp == "NeumannX_DirichletY")
            outletCondition_ = OutletCondition::neumannXdirichletY;
        else if (tmp == "NeumannX_NeumannY")
            outletCondition_ = OutletCondition::neumannXneumannY;
        else
            DUNE_THROW(Dune::InvalidStateException, tmp + " is not a valid outlet boundary condition");

        useVelocityProfile_ = getParam<bool>("Problem.UseVelocityProfile", false);
        outletPressure_ = getParam<Scalar>("Problem.OutletPressure", 1.1e5);
    }

   /*!
     * \name Problem parameters
     */
    // \{

   /*!
     * \brief Returns the temperature within the domain in [K].
     *
     * This problem assumes a temperature of 10 degrees Celsius.
     */
    Scalar temperature() const
    { return 273.15 + 10; } // 10C

    // \}
   /*!
     * \name Boundary conditions
     */
    // \{

   /*!
     * \brief Specifies which kind of boundary condition should be
     *        used for which equation on a given boundary control volume.
     *
     * \param globalPos The position of the center of the finite volume
     */
    BoundaryTypes boundaryTypesAtPos(const GlobalPosition &globalPos) const
    {
        BoundaryTypes values;



        // if(isInlet_(globalPos))
        // {
        //     if constexpr (Impl::isMomentumProblem<TypeTag>())
        //     {
        //         values.setDirichlet(Indices::velocityXIdx);
        //         values.setDirichlet(Indices::velocityYIdx);
        //     }
        //     else
        //     {
        //         values.setNeumann(0); // TODO idx
        //     }

        // }
        // else if(isOutlet_(globalPos))
        // {

        //     if constexpr (Impl::isMomentumProblem<TypeTag>())
        //     {
        //         // values.setNeumann(Indices::momentumXBalanceIdx);
        //         // values.setNeumann(Indices::momentumYBalanceIdx);
        //         values.setDirichlet(Indices::velocityXIdx);
        //         values.setDirichlet(Indices::velocityYIdx);
        //     }
        //     else
        //     {
        //         values.setNeumann(0); // TODO idx
        //         // values.setDirichlet(0); // TODO idx
        //     }

        // }
        // else
        // {
            if constexpr (ParentType::isMomentumProblem())
            {
                values.setDirichlet(Indices::velocityXIdx);
                values.setDirichlet(Indices::velocityYIdx);



                if (isOutlet_(globalPos))
                    values.setAllNeumann();
            }
            else
            {

                values.setNeumann(0);
            }
                // values.setDirichlet(0);

        // }

        return values;
    }


   /*!
     * \brief Evaluates the boundary conditions for a Dirichlet control volume.
     *
     * \param globalPos The center of the finite volume which ought to be set.
     */
    PrimaryVariables dirichletAtPos(const GlobalPosition &globalPos) const
    {
        PrimaryVariables values = initialAtPos(globalPos);

//         if(isInlet_(globalPos))
//         {
// #if NONISOTHERMAL
//             // give the system some time so that the pressure can equilibrate, then start the injection of the hot liquid
//             if(time() >= 200.0)
//                 values[Indices::temperatureIdx] = 293.15;
// #endif
//         }

        if constexpr (ParentType::isMomentumProblem())
        {
            values[Indices::velocityXIdx] = parabolicProfile(globalPos[1], inletVelocity_);
        }


        return values;
    }

    /*!
     * \brief Evaluates the boundary conditions for a Neumann control volume.
     *
     * \param element The element for which the Neumann boundary condition is set
     * \param fvGeometry The fvGeometry
     * \param elemVolVars The element volume variables
     * \param elemFaceVars The element face variables
     * \param scvf The boundary sub control volume face
     */
    template<class ElementVolumeVariables, class ElementFluxVariablesCache>
    NumEqVector neumann(const Element& element,
                        const FVElementGeometry& fvGeometry,
                        const ElementVolumeVariables& elemVolVars,
                        const ElementFluxVariablesCache& elemFluxVarsCache,
                        const SubControlVolumeFace& scvf) const
    {
        NumEqVector values(0.0);

        if constexpr (ParentType::isMomentumProblem())
        {
            // pressure contribution
            if (scvf.isFrontal())
            {
                values[scvf.directionIndex()] = outletPressure_;
                if constexpr (getPropValue<TypeTag, Properties::NormalizePressure>())
                    values[scvf.directionIndex()] -= outletPressure_;
            }

            // inertial terms
            if (this->enableInertiaTerms() && isOutlet_(scvf.ipGlobal()))
            {
                if (scvf.isFrontal())
                    values[Indices::momentumXBalanceIdx] += elemVolVars[scvf.insideScvIdx()].velocity() * elemVolVars[scvf.insideScvIdx()].velocity() * this->density(element, fvGeometry, scvf) * scvf.directionSign();

                else // scvf.isLateral()
                {
                    const auto transportingVelocity = [&]()
                    {
                        const auto& orthogonalScvf = fvGeometry.lateralOrthogonalScvf(scvf);
                        const auto innerTransportingVelocity = elemVolVars[orthogonalScvf.insideScvIdx()].velocity();

                        if (scvf.boundary())
                        {
                            if (const auto bcTypes = this->boundaryTypes(element, scvf); bcTypes.isDirichlet(scvf.directionIndex()))
                                return this->dirichlet(element, scvf)[scvf.directionIndex()];
                            else
                                return
                                    innerTransportingVelocity; // fallback
                        }
                        else
                        {
                            static const bool useOldScheme = getParam<bool>("FreeFlow.UseOldTransportingVelocity", true); // TODO how to deprecate?
                            if (useOldScheme)
                                return innerTransportingVelocity;
                            else
                            {
                                // average the transporting velocity by weighting with the scv volumes
                                const auto insideVolume = fvGeometry.scv(orthogonalScvf.insideScvIdx()).volume();
                                const auto outsideVolume = fvGeometry.scv(orthogonalScvf.outsideScvIdx()).volume();
                                const auto outerTransportingVelocity = elemVolVars[orthogonalScvf.outsideScvIdx()].velocity();
                                return (insideVolume*innerTransportingVelocity + outsideVolume*outerTransportingVelocity) / (insideVolume + outsideVolume);
                            }
                        }
                    }();

                    if (fvGeometry.scv(scvf.insideScvIdx()).boundary())
                    {
                        const auto innerVelocity = elemVolVars[scvf.insideScvIdx()].velocity();
                        const auto outerVelocity = elemVolVars[scvf.outsideScvIdx()].velocity();
                        const auto rho = this->getInsideAndOutsideDensity(element, fvGeometry, scvf);

                        const bool selfIsUpstream = scvf.directionSign() == sign(transportingVelocity);

                        const auto insideMomentum = innerVelocity * rho.first;
                        const auto outsideMomentum = outerVelocity * rho.second;

                        static const auto upwindWeight = getParamFromGroup<Scalar>(this->paramGroup(), "Flux.UpwindWeight");

                        const auto transportedMomentum =  selfIsUpstream ? (upwindWeight * insideMomentum + (1.0 - upwindWeight) * outsideMomentum)
                                                                         : (upwindWeight * outsideMomentum + (1.0 - upwindWeight) * insideMomentum);

                        values[Indices::momentumYBalanceIdx] += transportingVelocity * transportedMomentum * scvf.directionSign();
                    }
                    else
                    {
                        const auto insideDensity = this->density(element, fvGeometry.scv(scvf.insideScvIdx()));
                        const auto innerVelocity = elemVolVars[scvf.insideScvIdx()].velocity();
                        values[Indices::momentumYBalanceIdx] += innerVelocity * transportingVelocity * insideDensity * scvf.directionSign();
                    }
                }
            }

            // viscous terms
            if (outletCondition_ == OutletCondition::doNothing)
                values[Indices::momentumYBalanceIdx] = 0;
            else if (outletCondition_ == OutletCondition::outflow) // TODO put in outflow helper
            {
                if (scvf.isLateral() && !fvGeometry.scv(scvf.insideScvIdx()).boundary())
                {
                    const auto mu = this->effectiveViscosity(element, fvGeometry, scvf);
                    values[Indices::momentumYBalanceIdx] -= mu * StaggeredVelocityGradients::velocityGradJI(fvGeometry, scvf, elemVolVars) * scvf.directionSign();
                }

                if (scvf.isLateral() && fvGeometry.scv(scvf.insideScvIdx()).boundary())
                {
                    const auto mu = this->effectiveViscosity(element, fvGeometry, scvf);
                    values[Indices::momentumYBalanceIdx] -= mu * StaggeredVelocityGradients::velocityGradIJ(fvGeometry, scvf, elemVolVars) * scvf.directionSign();
                }
            }
            else
            {
                assert(outletCondition_ == OutletCondition::neumannXneumannY);
                values[Indices::momentumYBalanceIdx] = -dudy(scvf.ipGlobal()[1], inletVelocity_) * this->effectiveViscosity(element, fvGeometry, scvf) * scvf.directionSign();
            }
        }

        else
        {
            if (isInlet_(scvf.ipGlobal()) || isOutlet_(scvf.ipGlobal()))
            {
                const auto insideDensity = isInlet_(scvf.ipGlobal()) ? 1.35313 : elemVolVars[scvf.insideScvIdx()].density();
                values[Indices::conti0EqIdx] = this->faceVelocity(element, fvGeometry, scvf) * insideDensity * scvf.unitOuterNormal();
            }
        }

        return values;
    }

    /*!
     * \brief A parabolic velocity profile.
     *
     * \param y The position where the velocity is evaluated.
     * \param vMax The profile's maxmium velocity.
     */
    Scalar parabolicProfile(const Scalar y, const Scalar vMax) const
    {
        const Scalar yMin = this->gridGeometry().bBoxMin()[1];
        const Scalar yMax = this->gridGeometry().bBoxMax()[1];
        return  vMax * (y - yMin)*(yMax - y) / (0.25*(yMax - yMin)*(yMax - yMin));
    }

    /*!
     * \brief The partial dervivative of the horizontal velocity (following a parabolic profile for
     *         Stokes flow) w.r.t. to the y-coordinate (du/dy).
     *
     * \param y The position where the derivative is evaluated.
     * \param vMax The profile's maxmium velocity.
     */
    Scalar dudy(const Scalar y, const Scalar vMax) const
    {
        const Scalar yMin = this->gridGeometry().bBoxMin()[1];
        const Scalar yMax = this->gridGeometry().bBoxMax()[1];
        return vMax * (4.0*yMin + 4*yMax - 8.0*y) / ((yMin-yMax)*(yMin-yMax));
    }

    // \}

   /*!
     * \name Volume terms
     */
    // \{

   /*!
     * \brief Evaluates the initial value for a control volume.
     *
     * \param globalPos The global position
     */
    PrimaryVariables initialAtPos(const GlobalPosition& globalPos) const
    {
        PrimaryVariables values;

        if constexpr (ParentType::isMomentumProblem())
        {
            values[Indices::velocityYIdx] = 0.0;
            if (useVelocityProfile_)
                values[Indices::velocityXIdx] = parabolicProfile(globalPos[1], inletVelocity_);
            else
                values[Indices::velocityXIdx] = inletVelocity_;
        }
        else
        {
            values[Indices::pressureIdx] = outletPressure_;
        }

#if NONISOTHERMAL
        values[Indices::temperatureIdx] = 283.15;
#endif


        return values;
    }

    // \}
    void setTimeLoop(TimeLoopPtr timeLoop)
    {
        timeLoop_ = timeLoop;
        if(inletVelocity_ > eps_)
            timeLoop_->setCheckPoint({200.0, 210.0});
    }

    Scalar time() const
    {
        return timeLoop_->time();
    }


private:

    bool isInlet_(const GlobalPosition& globalPos) const
    {
        return globalPos[0] < eps_;
    }

    bool isOutlet_(const GlobalPosition& globalPos) const
    {
        return globalPos[0] > this->gridGeometry().bBoxMax()[0] - eps_;
    }

    static constexpr Scalar eps_=1e-6;
    Scalar inletVelocity_;
    Scalar outletPressure_;
    OutletCondition outletCondition_;
    bool useVelocityProfile_;
    TimeLoopPtr timeLoop_;
    std::shared_ptr<CouplingManager> couplingManager_;
};
} // end namespace Dumux

#endif