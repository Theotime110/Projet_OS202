#include <vector>
#include <iostream>
#include <random>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"
#include <chrono>
#include <mpi.h>

double t_ants = 0.0, t_evap = 0.0, t_updt = 0.0, t_tot = 0.0;
void advance_time( std::vector<position_t>& tab_pos,
                   std::vector<int>& tab_state,
                   std::vector<std::size_t>& tab_seed,
                   const fractal_land& land, pheronome& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::size_t& cpteur )
{
    std::chrono::time_point t0 = std::chrono::high_resolution_clock::now();
    for ( size_t i = 0; i < tab_pos.size(); ++i )
        advance(tab_pos[i], tab_state[i], tab_seed[i], phen, land, pos_food, pos_nest, cpteur);
    std::chrono::time_point t1 = std::chrono::high_resolution_clock::now();
    double* phen_data = phen.data();
    int phen_size = static_cast <int>(phen.data_size());
    MPI_Allreduce(MPI_IN_PLACE, phen_data, phen_size, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    phen.do_evaporation();
    std::chrono::time_point t2 = std::chrono::high_resolution_clock::now();
    MPI_Allreduce(MPI_IN_PLACE, phen_data, phen_size, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    phen.update();
    std::chrono::time_point t3 = std::chrono::high_resolution_clock::now();
    t_ants += std::chrono::duration<double, std::milli>(t1-t0).count();
    t_evap += std::chrono::duration<double, std::milli>(t2-t1).count();
    t_updt += std::chrono::duration<double, std::milli>(t3-t2).count();
    t_tot += std::chrono::duration<double, std::milli>(t3-t0).count();
}

int main(int nargs, char* argv[])
{

    MPI_Init(&nargs, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) SDL_Init( SDL_INIT_VIDEO );
    std::size_t seed = 26; // Graine pour la génération aléatoire ( reproductible )
    const int nb_ants = 5000; // Nombre de fourmis
    const int nb_ants_loc = nb_ants/world_size; // Nombre de fourmis par processus

    const double eps = 0.8;  // Coefficient d'exploration
    const double alpha=0.7; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation
    // Location du nid
    position_t pos_nest{256,256};
    // Location de la nourriture
    position_t pos_food{500,500};
    //const int i_food = 500, j_food = 500;    
    // Génération du territoire 512 x 512 ( 2*(2^8) par direction )
    //fractal_land land(8,2,1.,1024);
    fractal_land land(8,2,1.,1024);
    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    /* On redimensionne les valeurs de fractal_land de sorte que les valeurs
    soient comprises entre zéro et un */
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
            land(i,j) = (land(i,j)-min_val)/delta;
        }
    // Définition du coefficient d'exploration de toutes les fourmis.
    m_eps = eps;
    // On va créer des fourmis un peu partout sur la carte :
    //int debut_loc = world_rank*nb_ants_loc ;
    std::vector<position_t> tab_pos;
    std::vector<int> tab_state;
    std::vector<std::size_t> tab_seed;
    tab_pos.reserve(nb_ants);
    tab_state.reserve(nb_ants);
    tab_seed.reserve(nb_ants);
    auto gen_ant_pos = [&land, &seed] () { return rand_int32(0, land.dimensions()-1, seed); };
    for ( size_t i = 0; i < nb_ants_loc; ++i ){
        tab_pos.emplace_back(position_t{gen_ant_pos(),gen_ant_pos()});
        tab_seed.emplace_back(seed);
        tab_state.emplace_back(0);
        }
    std::vector<position_t> tab_pos_loc;
    std::vector<int> tab_state_loc;
    std::vector<std::size_t> tab_seed_loc;
    /*
    MPI_Datatype mpi_size_t;
    if (sizeof(size_t) == 4) {mpi_size_t = MPI_UINT32_T;}
    else mpi_size_t = {MPI_UINT64_T;}


    MPI_Scatter(tab_pos.data(), nb_ants_loc, MPI_2INT, &tab_pos_loc, nb_ants_loc, MPI_2INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(tab_state.data(), nb_ants_loc, MPI_INT, &tab_state_loc, nb_ants_loc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(tab_seed.data(), nb_ants_loc, mpi_size_t, &tab_seed_loc, nb_ants_loc, mpi_size_t, 0, MPI_COMM_WORLD);
    */

    // On crée toutes les fourmis dans la fourmilière.
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);
    if (world_rank == 0){
        Window win("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
        Renderer renderer( land, phen, pos_nest, pos_food, tab_pos );
    }
    // Compteur de la quantité de nourriture apportée au nid par les fourmis
    size_t food_quantity = 0;
    size_t food_local = 0;
    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;
    Window*   win      = nullptr;
    Renderer* renderer = nullptr;
    if (world_rank == 0){
        win = new Window ("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
        renderer = new Renderer( land, phen, pos_nest, pos_food, tab_pos );
    }
    while (cont_loop) {
        ++it;
        if (world_rank == 0){
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    cont_loop = false;
            }
        }   

        advance_time(tab_pos, tab_state, tab_seed, land, phen, pos_nest, pos_food, food_local );
        if (world_rank == 0){
            if ( it % 100 == 0 ) {
               std::cout << "Itération :" << it << std::endl;
                std::cout << "  Fourmis     : " << t_ants   << " ms\n";
                std::cout << "  Evaporation : " << t_evap   << " ms\n";
                std::cout << "  Update      : " << t_updt << " ms\n";
                std::cout << "  Total       : " << t_tot << " ms\n";
                if (it == 1000){
                    std::cout << "  Ratio fourmis     : " << t_ants/t_tot << "\n";
                    std::cout << "  Ratio evaporation : " << t_evap/t_tot << "\n";
                    std::cout << "  Ratio update      : " << t_updt/t_tot << "\n";

                }
            }
        }
        MPI_Reduce(&food_local, &food_quantity, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD );

        if (world_rank == 0){
            renderer->display( *win, food_quantity );
            win->blit();
        }
        if ( not_food_in_nest && food_quantity > 0 ) {
            std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
            not_food_in_nest = false;
        }
        //SDL_Delay(10);
    }
    SDL_Quit();
    return 0;
}