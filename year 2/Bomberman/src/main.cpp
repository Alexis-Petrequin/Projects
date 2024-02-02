#include "graphic.hpp"

int main(void)
{
    InitWindow(1920, 1080, "Bomberman");
    InitAudioDevice();
    Game *game = new Game();
    int mode = 3;

    srand((unsigned int)time(NULL));
    SetTargetFPS(60);
    SetExitKey(0);
    while (!WindowShouldClose() && mode) {
        if (mode == 1) {
            game->restart();
            mode = game->play(game->_nb_player, game->_nb_ia, createMap());
        }
        if (mode == 2) {
            game->restart();
            mode = 1;
        }
        if (mode == 3)
            mode = game->menu();
        if (mode == 4) {
            game->restart();
            game->load();
            mode = game->play(game->_nb_player, game->_nb_ia, game->_map);
        }

    }
    delete game;
    CloseAudioDevice();
    CloseWindow();
    return 0;
}