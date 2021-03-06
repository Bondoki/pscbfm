#pragma once

#ifndef INLINE_GRAPHCOLORING
#   define INLINE_GRAPHCOLORING inline
#endif

#include "graphColoring.h"

#include <algorithm>                    // sort, swap
#include <cassert>
#include <cstdint>                      // uint8_t
#include <iostream>
#include <limits>                       // max
#include <map>
#include <functional>
#include <utility>                      // pair
#include <vector>
#include <stack>



namespace { // anonymous namespace with functions only to be used inside this file

//SelectedLog mLog = SelectedLog( __FILENAME__ );

/**
 * Sort v[0] <= v[1] <= ... <= v[i] using Bubble sort
 */
template< typename T_RandomIt, typename T_Comparator >
inline void bubbleSort
(
    T_RandomIt   const & i0,
    T_RandomIt   const & i1,
    T_Comparator const & cmp // = operator< < decltype(*i0) > ... how to do this?
)
{
    T_RandomIt iFirstSorted = i1;
    while ( iFirstSorted != i0 )
    {
        T_RandomIt const iLastFirstSorted = iFirstSorted;
        iFirstSorted = i0;
        T_RandomIt jp1 = i0; ++jp1;
        for ( T_RandomIt j = i0; jp1 != iLastFirstSorted; ++j, ++jp1 )
        {
            if ( ! cmp( *j, *jp1 ) )
            {
                std::swap( *j, *jp1 );
                iFirstSorted = jp1;
            }
        }
    }
}

template< typename T, typename T_Comparator >
inline
void bubbleSort( std::vector< T > & v, T_Comparator const & cmp )
{
    auto nUnsorted = v.size();
    while ( nUnsorted > 1 )
    {
        size_t iLastSwap = 0;
        for ( size_t j = 0u; j < nUnsorted-1; ++j )
        {
            if ( ! cmp( v[j], v[j+1] ) )
            {
                std::swap( v[j], v[j+1] );
                iLastSwap = j;
            }
        }
        nUnsorted = iLastSwap+1;
    }
}

} // anonymous namespace


template< class T_Neighbors, typename T_Id, typename T_Color >
INLINE_GRAPHCOLORING
std::vector< T_Color > graphColoring
(
    T_Neighbors const & rvNeighbors,
    size_t      const & rnElements ,
    bool        const   rbUniformColors,
    std::function< size_t( T_Neighbors const &, T_Id const & ) > const & rfGetNeighborsSize,
    std::function< T_Id( T_Neighbors const &, T_Id const &, size_t const & ) > const & rfGetNeighbor
)
{
    T_Color const cNoColor = std::numeric_limits< T_Color >::max();
    /* for int32_t we would allocate 2e9 sized vector of ints for counting each color ... That's why this  */
    T_Color const nMaxColors = std::min( (T_Color) 256, std::numeric_limits< T_Color >::max() ) - 1;

    /* automatic coloring routine */
    std::vector< T_Color > vColors( rnElements, cNoColor ); // the end result, which is to be returned
    std::map< T_Color, bool > vbColorUsed; // In each loop this will be used to check which colors can still be assigned to the current node. Note that default constructor for bool is false

    std::vector< size_t > nColorsUsed; // count how many times each color gets assigned, note that default constructor for size_t is 0. Needed for uniform coloring
    for ( size_t iNode = 0u; iNode < rnElements; ++iNode )
    {
        /* skip over colored monomers */
        if ( vColors.at( iNode ) != cNoColor )
            continue;

        /* here we want to color the whole connected polymer by following its
         * connections. This means, that if it has multiple connections we need
         * to save backtracking ids to resume traversing until the polymer is
         * colored
         * Alternatively we could just skip that and instead just let the outer
         * loop over all monomers do this work. But that would lead to less
         * deterministic results, as not only the first appearing monomer in
         * mPolymerSystem influences the whole polymer coloring, but basically
         * every monomer after a fork can lead to different colorings if it
         * has a different sorting inside mPolymersystem */
        std::stack< uint32_t > iNodesTodo;
        long long int iNodeIsolatedGraph = iNode;
        do
        {
            /* reset vbColorUsed map */
            for ( auto & bColorUsed : vbColorUsed )
                bColorUsed.second = false;

            size_t nUncoloredNeighbors = 0; // needed as break condition and for deciding what to do next
            long long int const cNoNode = -1;
            auto iFirstUncoloredNeighbor = cNoNode; // will be worked on next if there is one. Note that this can't be zero, because we started this loop by coloring node 0. This is the global monomer ID, not the ID inside the neighbor list!
            assert( iNodeIsolatedGraph >= 0 );
            assert( (size_t) iNodeIsolatedGraph < rnElements );
            /* traverse over neighbors and check used colors */
            for ( size_t iNeighbor = 0; iNeighbor < rfGetNeighborsSize( rvNeighbors, iNodeIsolatedGraph ); ++iNeighbor )
            {
                T_Id const idNeighbor = rfGetNeighbor( rvNeighbors, iNodeIsolatedGraph, iNeighbor );
                auto const c = vColors.at( idNeighbor );
                /* this is for the backtracking later */
                if ( c != cNoColor )
                    vbColorUsed[c] = true;
                else
                {
                    ++nUncoloredNeighbors;
                    if ( iFirstUncoloredNeighbor == cNoNode )
                        iFirstUncoloredNeighbor = idNeighbor;
                }
            }

            /* find first unused color */
            long long int iColor = 0;
            /* Note that return of size() might change while doing the loop if a new color was inserted */
            while ( iColor < (long long int) vbColorUsed.size() && vbColorUsed[ iColor ] && iColor < nMaxColors )
                ++iColor;
            if ( iColor == nMaxColors )
                throw std::runtime_error( "Already used up the maximum amount of colors. Can't color the given polymer system automatically!" );
            /* Assign / apply color! */
            vColors.at( iNodeIsolatedGraph ) = iColor;
            if ( (size_t) iColor >= nColorsUsed.size() )
                nColorsUsed.resize( iColor+1, 0 );
            nColorsUsed[ iColor ]++;

            /* advance to next neighbor. We might need to backtrack */
            /* end when nothing to do anymore */
            if ( nUncoloredNeighbors == 0 && iNodesTodo.empty() )
                break;
            /* push to stack, that even when following one neighbor on this node
             * that there is still work to be done */
            if ( nUncoloredNeighbors > 1 )
                iNodesTodo.push( iNodeIsolatedGraph );

            /* iNodeIsolatedGraph might be cNoNode, but in that case next if clause will be executed */
            iNodeIsolatedGraph = iFirstUncoloredNeighbor;
            #ifndef NDEBUG
                if ( nUncoloredNeighbors == 0 ) assert( iNodeIsolatedGraph == cNoNode );
                if ( iNodeIsolatedGraph == cNoNode ) assert( nUncoloredNeighbors == 0 );
            #endif
            if ( nUncoloredNeighbors == 0 )
            {
                /* Beware! Because of cycles in the graph it can happen that a former
                 * todo monomer might not have any colored neighbors anymore! */
                while ( ! iNodesTodo.empty() )
                {
                    iNodeIsolatedGraph = iNodesTodo.top();
                    assert( vColors[ iNodeIsolatedGraph ] != cNoColor );
                    /* find first uncolored neighbor, there might not be one
                     * anymore, but if there is and there are still more
                     * uncolored besides that one, then don't pop yet from list */
                    /* unfortunately this is almost the same as above, but here
                     * we don't need to find the first color available for
                     * coloring ... */
                    size_t nUncoloredNeighbors2 = 0;
                    auto iFirstUncoloredNeighbor2 = cNoNode;
                    for ( size_t iNeighbor = 0; iNeighbor <
                    rfGetNeighborsSize( rvNeighbors, iNodeIsolatedGraph ); ++iNeighbor )
                    {
                        auto const idNeighbor = rfGetNeighbor( rvNeighbors, iNodeIsolatedGraph, iNeighbor );
                        if ( vColors.at( idNeighbor ) == cNoColor )
                        {
                            ++nUncoloredNeighbors2;
                            if ( iFirstUncoloredNeighbor2 == cNoNode )
                                iFirstUncoloredNeighbor2 = idNeighbor;
                        }
                    }
                    if ( nUncoloredNeighbors2 < 2 )
                        iNodesTodo.pop();
                    if ( iFirstUncoloredNeighbor2 != cNoNode ) // <=> nUncoloredNeighbors2 > 0
                    {
                        iNodeIsolatedGraph = iFirstUncoloredNeighbor2;
                        break;
                    }
                    iNodeIsolatedGraph = cNoNode;
                }
                if ( iNodesTodo.empty() && iNodeIsolatedGraph == cNoNode )
                    break; // whole coloring loop. We are finished
            }
            assert( iNodeIsolatedGraph != cNoNode );
        }
        while ( true );
    }

    /* if wished, do a second run flipping the most frequent colors to the
     * least frequent colors, until we reached unform distribution */
    if ( rbUniformColors )
    {
        std::vector< std::pair< T_Color, size_t > > nColorsUsedRarestFirst( nColorsUsed.size() );
        for ( size_t i = 0u; i < nColorsUsed.size(); ++i )
        {
            nColorsUsedRarestFirst[i].first  = i;
            nColorsUsedRarestFirst[i].second = nColorsUsed[i];
        }
        std::sort( nColorsUsedRarestFirst.begin(), nColorsUsedRarestFirst.end(),
                   []( std::pair< T_Color, size_t > const & a,
                       std::pair< T_Color, size_t > const & b )
                   { return a.second < b.second; } );
        size_t const nTargetFreq = ( rnElements + nColorsUsed.size() - 1 ) / nColorsUsed.size(); // ceil div
        for ( size_t iNode = 0u; iNode < rnElements; ++iNode )
        {
            if ( nColorsUsed[ vColors[iNode] ] <= nTargetFreq )
                continue;

            /* find rarest color which can be assigned */
            for ( size_t i = 0u; i < nColorsUsedRarestFirst.size() &&
                  nColorsUsedRarestFirst[i].second < nTargetFreq; ++i )
            {
                T_Color const newColor = nColorsUsedRarestFirst[i].first;
                /* check neighbors */
                bool changeColor = true;
                for ( size_t iNeighbor = 0; iNeighbor <
                rfGetNeighborsSize( rvNeighbors, iNode ); ++iNeighbor )
                {
                    auto const idNeighbor = rfGetNeighbor( rvNeighbors, iNode, iNeighbor );
                    if ( vColors.at( idNeighbor ) == newColor )
                        changeColor = false;
                }
                /* if we can set the current iNode to a less than average
                 * color, then do it now and keep lists up to date */
                if ( changeColor )
                {
                    T_Color const oldColor = vColors[ iNode ];
                    assert( oldColor != newColor && "The new rarer than average color is actually the same as the too frequent color! This should not happen." );
                    --nColorsUsed[ oldColor ];
                    ++nColorsUsed[ newColor ];
                    size_t iNew = 0, iOld = 0;
                    for ( size_t i = 0u; i < nColorsUsedRarestFirst.size(); ++i )
                    {
                        if ( nColorsUsedRarestFirst[i].first == oldColor )
                            iOld = i;
                        if ( nColorsUsedRarestFirst[i].first == newColor )
                            iNew = i;
                    }
                    assert( nColorsUsedRarestFirst.at( iOld ).first == oldColor && "old color not found in nColorsUsedRarestFirst!" );
                    assert( nColorsUsedRarestFirst.at( iNew ).first == newColor && "old color not found in nColorsUsedRarestFirst!" );
                    --nColorsUsedRarestFirst[ iOld ].second;
                    ++nColorsUsedRarestFirst[ iNew ].second;
                    /**
                     * resort nColorsUsedRarestFirst. Bubble sort is without
                     * question the best for that, because:
                     *   1. array was sorted
                     *   2. we change only two elements and only by +1 or -1
                     *   3. 1 && 2 => e.g. if all entries are different and
                     *      those two elements are not neighbors, then at worst
                     *      only two swaps are necessary and only one loop
                     *      through the array
                     *   4. From 3 excluded border cases aren't that bad either
                     */
                    bubbleSort( nColorsUsedRarestFirst.begin(),
                                nColorsUsedRarestFirst.end(),
                                []( std::pair< T_Color, size_t > const & a,
                                    std::pair< T_Color, size_t > const & b )
                                { return a.second <= b.second; } );

                    /* actually change color */
                    vColors[ iNode ] = newColor;
                }
            }
        }
    }

    return vColors;
}


#undef INLINE_GRAPHCOLORING
