#include "ant.hpp"
#include <iostream>
#include "rand_generator.hpp"

double m_eps = 0.;

void advance( position_t& pos, int& state, std::size_t& seed, pheronome& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest,
                   std::size_t& cpteur_food ) 
{
    auto ant_choice = rand_double( 0., 1., seed );
    auto dir_choice = rand_int32( 1, 4, seed );
    double consumed_time = 0.;
    // Tant que la fourmi peut encore bouger dans le pas de temps imparti
    while ( consumed_time < 1. ) {
        // Si la fourmi est chargée, elle suit les phéromones de deuxième type, sinon ceux du premier.
        int        ind_pher    = state;
        double     choix       = rand_double( 0., 1., seed );
        position_t old_pos_ant = pos;
        position_t new_pos_ant = old_pos_ant;
        double max_phen    = std::max( {phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher],
                                     phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher],
                                     phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher],
                                     phen( new_pos_ant.x, new_pos_ant.y + 1 )[ind_pher]} );
        if ( ( choix > m_eps ) || ( max_phen <= 0. ) ) {
            do {
                new_pos_ant = old_pos_ant;
                int d = rand_int32( 1, 4, seed );
                if ( d==1 ) new_pos_ant.x  -= 1;
                if ( d==2 ) new_pos_ant.y -= 1;
                if ( d==3 ) new_pos_ant.x  += 1;
                if ( d==4 ) new_pos_ant.y += 1;

            } while ( phen[new_pos_ant][ind_pher] == -1 );
        } else {
            // On choisit la case où le phéromone est le plus fort.
            if ( phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher] == max_phen )
                new_pos_ant.x -= 1;
            else if ( phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher] == max_phen )
                new_pos_ant.x += 1;
            else if ( phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher] == max_phen )
                new_pos_ant.y -= 1;
            else  // if (phen(new_pos_ant.first,new_pos_ant.second+1)[ind_pher] == max_phen)
                new_pos_ant.y += 1;
        }
        consumed_time += land( new_pos_ant.x, new_pos_ant.y);
        phen.mark_pheronome( new_pos_ant );
        pos = new_pos_ant;
        if ( pos == pos_nest ) {
            if ( state ) {
                cpteur_food += 1;
            }
            state = 0;
        }
        if ( pos == pos_food ) {
            state = 1;
        }
    }
}