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
/** \file
  * \brief This file contains different higher order methods for approximating the velocity.
  */

#ifndef DUMUX_HIGHER_ORDER_VELOCITY_APPROXIMATION_HH
#define DUMUX_HIGHER_ORDER_VELOCITY_APPROXIMATION_HH

#include <cmath>
#include <functional>
#include <iostream>

namespace Dumux {

/**
  * \brief This file contains different higher order methods for approximating the velocity.
  */
template<class Scalar>
class HigherOrderApproximation
{
public:
    /**
      * \brief Upwind Method
      */
    Scalar upwind(const Scalar downstreamVelocity,
                  const Scalar upstreamVelocity,
                  const Scalar density) const
    {
        return upstreamVelocity * density;
    }

    /**
      * \brief Central Differencing Method
      */
    Scalar centralDifference(const Scalar downstreamVelocity,
                             const Scalar upstreamVelocity,
                             const Scalar density) const
    {
        return (0.5 * (upstreamVelocity + downstreamVelocity)) * density;
    }

    /**
      * \brief Linear Upwind Method
      */
    Scalar linearUpwind(const Scalar upstreamVelocity,
                        const Scalar upUpstreamVelocity,
                        const Scalar upstreamToDownstreamDistance,
                        const Scalar upUpstreamToUpstreamDistance,
                        const Scalar density) const
    {
        Scalar zeroOrder = upstreamVelocity;
        Scalar firstOrder = -1.0 * ((upstreamVelocity - upUpstreamVelocity) / upUpstreamToUpstreamDistance) * ( upstreamToDownstreamDistance / -2.0);
        return (zeroOrder + firstOrder) * density;
    }

    /**
      * \brief QUICK upwinding Scheme: Quadratic Upstream Interpolation for Convective Kinematics
      */
    Scalar upwindQUICK(const Scalar downstreamVelocity,
                       const Scalar upstreamVelocity,
                       const Scalar upUpstreamVelocity,
                       const Scalar upstreamToDownstreamDistance,
                       const Scalar upUpstreamToUpstreamDistance,
                       const Scalar density) const
    {
        Scalar normalDistance = (upUpstreamToUpstreamDistance + upstreamToDownstreamDistance) / 2.0;
        Scalar zeroOrder = upstreamVelocity;
        Scalar firstOrder = ((downstreamVelocity - upstreamVelocity) / 2.0);
        Scalar secondOrder = -(((downstreamVelocity - upstreamVelocity) / upstreamToDownstreamDistance) - ((upstreamVelocity - upUpstreamVelocity) / upUpstreamToUpstreamDistance))
                           * upstreamToDownstreamDistance * upstreamToDownstreamDistance / (8.0 * normalDistance);
        return (zeroOrder + firstOrder + secondOrder) * density;
    }

    /**
      * \brief TVD Scheme: Total Variation Diminuishing
      */
    Scalar TVD(const Scalar downstreamVelocity,
               const Scalar upstreamVelocity,
               const Scalar upUpstreamVelocity,
               const Scalar density,
               const std::function<Scalar(const Scalar)>& limiter) const
    {
        const Scalar ratio = (upstreamVelocity - upUpstreamVelocity) / (downstreamVelocity - upstreamVelocity);

        // If the velocity field is uniform (like at the first newton step) we get a NaN
        if(ratio > 0.0 && std::isfinite(ratio))
        {
            const Scalar secondOrderTerm = 0.5 * limiter(ratio) * (downstreamVelocity - upstreamVelocity);
            return density * (upstreamVelocity + secondOrderTerm);
        }

        else
            return density * upstreamVelocity;
    }

    /**
      * \brief TVD Scheme: Total Variation Diminuishing
      *
      * This functions manages the non uniformities of the grid according to [Li, Liao 2007].
      * It tries to reconstruct the value for the velocity at the upstream-upstream point
      * if the grid was uniform.
      */
    Scalar TVD(const Scalar downstreamVelocity,
               const Scalar upstreamVelocity,
               const Scalar upUpstreamVelocity,
               const Scalar upstreamToDownstreamDistance,
               const Scalar upUpstreamToUpstreamDistance,
               const bool selfIsUpstream,
               const Scalar density,
               const std::function<Scalar(const Scalar)>& limiter) const
    {
        // I need the information of selfIsUpstream to get the correct sign because upUpstreamToUpstreamDistance is always positive
        const Scalar upUpstreamGradient = (upstreamVelocity - upUpstreamVelocity) / upUpstreamToUpstreamDistance * selfIsUpstream;

        // Distance between the upUpstream node and the position where it should be if the grid were uniform.
        const Scalar correctionDistance = upUpstreamToUpstreamDistance - upstreamToDownstreamDistance;
        const Scalar reconstrutedUpUpstreamVelocity = upUpstreamVelocity + upUpstreamGradient * correctionDistance;
        const Scalar ratio = (upstreamVelocity - reconstrutedUpUpstreamVelocity) / (downstreamVelocity - upstreamVelocity);

        // If the velocity field is uniform (like at the first newton step) we get a NaN
        if(ratio > 0 && std::isfinite(ratio))
        {
            const Scalar secondOrderTerm = 0.5 * limiter(ratio) * (downstreamVelocity - upstreamVelocity);
            return density * (upstreamVelocity + secondOrderTerm);
        }

        else
            return density * upstreamVelocity;
    }

    /**
     * \brief TVD Scheme: Total Variation Diminuishing
     *
     * This functions manages the non uniformities of the grid according to [Hou, Simons, Hinkelmann 2007].
     * It should behave better then the Li's version in very stretched grids.
     */
    Scalar TVD(const Scalar downstreamVelocity,
               const Scalar upstreamVelocity,
               const Scalar upUpstreamVelocity,
               const Scalar upstreamToDownstreamDistance,
               const Scalar upUpstreamToUpstreamDistance,
               const Scalar downstreamStaggeredCellSize,
               const Scalar density,
               const std::function<Scalar(const Scalar, const Scalar)>& limiter) const
    {
        const Scalar ratio = (upstreamVelocity - upUpstreamVelocity) / (downstreamVelocity - upstreamVelocity)
                           * upstreamToDownstreamDistance / upUpstreamToUpstreamDistance;

        // If the velocity field is uniform (like at the first newton step) we get a NaN
        if(ratio > 0.0 && std::isfinite(ratio))
        {
            const Scalar upstreamStaggeredCellSize = 0.5 * (upstreamToDownstreamDistance + upUpstreamToUpstreamDistance);
            const Scalar Rfactor = (upstreamStaggeredCellSize + downstreamStaggeredCellSize) / upstreamStaggeredCellSize;
            const Scalar secondOrderTerm = limiter(ratio, Rfactor) / Rfactor * (downstreamVelocity - upstreamVelocity);
            return density * (upstreamVelocity + secondOrderTerm);
        }
        else
            return density * upstreamVelocity;
    }

    /**
      * \brief Van Leer flux limiter function [Van Leer 1974]
      */
    static Scalar vanleer(const Scalar r)
    {
        return 2.0 * r / (1.0 + r);
    }

    /**
      * \brief Van Albada flux limiter function [Van Albada et al. 1982]
      */
    static Scalar vanalbada(const Scalar r)
    {
        return r * (r + 1.0) / (1.0 + r * r);
    }

    /**
      * \brief MinMod flux limiter function [Roe 1985]
      */
    static Scalar minmod(const Scalar r)
    {
        return std::min(r, 1.0);
    }

    /**
      * \brief SUPERBEE flux limiter function [Roe 1985]
      */
    static Scalar superbee(const Scalar r)
    {
        return std::max(std::min(2.0 * r, 1.0), std::min(r, 2.0));
    }

    /**
      * \brief UMIST flux limiter function [Lien and Leschziner 1993]
      */
    static Scalar umist(const Scalar r)
    {
        return std::min({2.0 * r, (1.0 + 3.0 * r) / 4.0, (3.0 + r) / 4.0, 2.0});
    }

    /*
     * \brief Monotonized-Central limiter [Van Leer 1977]
     */
    static Scalar mclimiter(const Scalar r)
    {
        return std::min({2.0 * r, (r + 1.0) / 2.0, 2.0});
    }

    /**
      * \brief Modified Van Leer flux limiter function [Hou, Simons, Hinkelmann 2007]
      */
    static Scalar modifiedVanleer(const Scalar r, const Scalar Rfactor)
    {
        return Rfactor * r / (Rfactor - 1.0 + r);
    }

    /**
      * \brief Modified SUPERBEE flux limiter function [Hou, Simons, Hinkelmann 2007]
      */
    static Scalar modifiedSuperbee(const Scalar r, const Scalar Rfactor)
    {
        return std::max(std::min(Rfactor * r, 1.0), std::min(r, Rfactor));
    }

    /**
      * \brief WAHYD Scheme [Hou, Simons, Hinkelmann 2007];
      */
    static Scalar wahyd(const Scalar r, const Scalar Rfactor)
    {
        return r > 1 ? std::min((r + Rfactor * r * r) / (Rfactor + r * r), Rfactor)
                     : modifiedVanleer(r, Rfactor);
    }

};

} // end namespace Dumux

#endif // DUMUX_HIGHER_ORDER_VELOCITY_APPROXIMATION_HH
