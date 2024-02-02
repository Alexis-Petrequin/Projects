#include "graphic.hpp"

void Game::draw_3D()
{
    Color color[4] = { WHITE, { 230, 33, 55, 255 }, GREEN, BLUE };

    draw_map();
    for (int i = 0; i < _nb_player + _nb_ia; i++)
        if (_character[i]._live > 0)
            DrawModelEx(_character[i].model, _character[i].character_pos, { 1.0f, 0.0f, 0.0f }, -90.0f, { 0.01f, 0.01f, 0.01f }, color[i]);
    draw_bomb();
    draw_fire();
    draw_bonus();
}

void Game::draw_map()
{
    std::map<Vector2,char>::iterator it = _map.begin();

    DrawCubeTexture(_texture[2], {6.00f, 0.40f, 4.00f}, 27.00f, 0.00f, 16.00f, WHITE);       //draw l'arriere plan
    for (it = _map.begin(); it != _map.end(); ++it) {
        DrawCubeTexture(_texture[0], { it->first.x, 0.0f, it->first.y }, CUBEDIM, WHITE);    //draw le sol
        if (it->second == '*')                                                              //draw les murs
            DrawCubeTexture(_texture[4], { it->first.x, 1.00f, it->first.y }, CUBEDIM, WHITE);
        if (it->second == '&')                                                              // draw les obstacle
            DrawCubeTexture(_texture[3], { it->first.x, 1.00f, it->first.y }, 0.90f, 0.90f, 0.90f, WHITE);
        if (it->second == 'b')                                                              // draw les bordures
            DrawCubeTexture(_texture[1], { it->first.x, 1.00f, it->first.y }, 1.00f, 1.20f, 1.00f, WHITE);
    }
}

void Game::draw_bomb()
{
    std::vector<Bomb>::iterator it;
    int time = 0;

    for (it = _bomb.begin(); it != _bomb.end(); ++it) {
        if (_mode == 1)
            time = (clock() - it->_tcreated) * 1000 / CLOCKS_PER_SEC;
        else
            time = ((int)(_time - it->_tcreated) * 1000) / CLOCKS_PER_SEC;
        if (it->_state == 'a') {
            draw_radius_left(*it, 'X');
            draw_radius_up(*it, 'X');
            draw_radius_right(*it, 'X');
            draw_radius_down(*it, 'X');
            _map.find({it->_pos.x, it->_pos.z})->second = 'o';
        }
        if (it->_state == 'a' && time < 350) {          //fait clignoter la bombe
            DrawSphere(it->_pos, 0.30f, DARKGRAY);
            continue;
        }
        if (it->_state == 'a' && time < 700) {
            DrawSphere(it->_pos, 0.30f, RED);
            continue;
        }
        if (it->_state == 'a' && time < 1050) {
            DrawSphere(it->_pos, 0.30f, DARKGRAY);
            continue;
        }
        if (it->_state == 'a' && time < 1400) {
            DrawSphere(it->_pos, 0.30f, RED);
            continue;
        }
        if (it->_state == 'a' && time < 1750) {
            DrawSphere(it->_pos, 0.30f, DARKGRAY);
            continue;
        }
        if (it->_state == 'a' && time < 2000) {
            DrawSphere(it->_pos, 0.30f, RED);
            continue;
        }
        if (it->_state == 'a' && time >= 2000) {    //passer l'etat de 'a' à 'b' fait exploser la bombe
            it->_state = 'b';
            it->_tcreated = clock();                // reset du timer pour s'en servir pour calcuer la durée d'explosion
            it->radius[0] = radius_left(*it);  //determine la range de l'explosion en prenant en compte la collision blocs
            it->radius[1] = radius_up(*it);
            it->radius[2] = radius_right(*it);
            it->radius[3] = radius_down(*it);
            it->character->_nbbomb++;
            PlaySoundMulti(_sound[0]);
        }
    }
}

void Game::draw_fire()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    int time = 0;

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b') {
            if (_mode == 1)
                time = (clock() - it->_tcreated) * 1000 / CLOCKS_PER_SEC;
            else
                time = ((int)(_time - it->_tcreated) * 1000) / CLOCKS_PER_SEC;
            if (time < 300)
                for (int i = 0; i < it->_power; i++) {
                    if (i < it->radius[0]) {
                        DrawCubeTexture(_texture[5], { it->_pos.x - i, 1.0f, it->_pos.z}, 1.00f, 1.00f, 1.00f, WHITE);
                        _map.find({it->_pos.x - i, it->_pos.z})->second = '!';
                    }
                    if (i < it->radius[1]) {
                        DrawCubeTexture(_texture[5], { it->_pos.x, 1.0f, it->_pos.z - i}, 1.00f, 1.00f, 1.00f, WHITE);
                        _map.find({it->_pos.x, it->_pos.z - i})->second = '!';
                    }
                    if (i < it->radius[2]) {
                        DrawCubeTexture(_texture[5], { it->_pos.x + i, 1.0f, it->_pos.z}, 1.00f, 1.00f, 1.00f, WHITE);
                        _map.find({it->_pos.x + i, it->_pos.z})->second = '!';
                    }
                    if (i < it->radius[3]) {
                        DrawCubeTexture(_texture[5], { it->_pos.x, 1.0f, it->_pos.z + i}, 1.00f, 1.00f, 1.00f, WHITE);
                        _map.find({it->_pos.x, it->_pos.z + i})->second = '!';
                    }
                } else {
                    draw_radius_left(*it, ' ');
                    draw_radius_up(*it, ' ');
                    draw_radius_right(*it, ' ');
                    draw_radius_down(*it, ' ');
                    _map.find({it->_pos.x, it->_pos.z})->second = ' ';
                    _bomb.erase(it);
                    --it;
                }
        }
}

void Game::draw_bonus()
{
    std::vector<Bonus>::iterator it = _bonus.begin();

    for (it = _bonus.begin(); it != _bonus.end(); ++it) {
        it->_pos.y += it->_up;
        if (it->_pos.y > 1.00f && it->_up == 0.01f)
            it->_up = -0.01f;
        if (it->_pos.y < 0.6f && it->_up == -0.01f)
            it->_up = 0.01f;
        if (it->_type == '1') {
            DrawCubeTexture(_texture[6], it->_pos, 0.60f, 0.0f, 0.60f, WHITE);
            if ( (it->_pos.y > 0.58 && it->_pos.y < 0.65) || (it->_pos.y > 0.75 && it->_pos.y < 0.85) || (it->_pos.y > 0.95 && it->_pos.y < 1.02)) { //fait clignoter le contour jaune des bonus
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.61f, 0.0f, 0.61f, YELLOW);  //permet d'ajouter un contour jaune aux bonus
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.62f, 0.0f, 0.62f, YELLOW);
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.63f, 0.0f, 0.63f, YELLOW);
            }
        }
        if (it->_type == '2') {
            DrawCubeTexture(_texture[7], it->_pos, 0.60f, 0.0f, 0.60f, WHITE);
            if ( (it->_pos.y > 0.58 && it->_pos.y < 0.65) || (it->_pos.y > 0.75 && it->_pos.y < 0.85) || (it->_pos.y > 0.95 && it->_pos.y < 1.02)) {
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.61f, 0.0f, 0.61f, YELLOW);
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.62f, 0.0f, 0.62f, YELLOW);
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.63f, 0.0f, 0.63f, YELLOW);
            }
        }
        if (it->_type == '3') {
            DrawCubeTexture(_texture[8], it->_pos, 0.60f, 0.0f, 0.60f, WHITE);
            if ( (it->_pos.y > 0.58 && it->_pos.y < 0.65) || (it->_pos.y > 0.75 && it->_pos.y < 0.85) || (it->_pos.y > 0.95 && it->_pos.y < 1.02)) {
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.61f, 0.0f, 0.61f, YELLOW);
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.62f, 0.0f, 0.62f, YELLOW);
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.63f, 0.0f, 0.63f, YELLOW);
            }
        }
        if (it->_type == '4') {
            DrawCubeTexture(_texture[9], it->_pos, 0.60f, 0.0f, 0.60f, WHITE);
            if ( (it->_pos.y > 0.58 && it->_pos.y < 0.65) || (it->_pos.y > 0.75 && it->_pos.y < 0.85) || (it->_pos.y > 0.95 && it->_pos.y < 1.02)) {
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.61f, 0.0f, 0.61f, YELLOW);
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.62f, 0.0f, 0.62f, YELLOW);
                DrawCubeWires( { it->_pos.x, it->_pos.y, it->_pos.z }, 0.63f, 0.0f, 0.63f, YELLOW);
            }
        }
    }
}