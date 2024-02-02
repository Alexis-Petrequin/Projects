#include "graphic.hpp"

void Game::fire_col_bomb_left()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    std::vector<Bomb>::iterator it2 = _bomb.begin();

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (it2 = _bomb.begin(); it2 != _bomb.end(); ++it2)
                if (it2->_state == 'a')
                    for (int i = 0; i < it->radius[0]; i++)
                        if (it2->_pos.x == it->_pos.x - i && it2->_pos.z == it->_pos.z) {
                            it2->_state = 'b';
                            it2->_tcreated = clock();
                            it2->radius[0] = radius_left(*it2);
                            it2->radius[1] = radius_up(*it2);
                            it2->radius[2] = radius_right(*it2);
                            it2->radius[3] = radius_down(*it2);
                            it2->character->_nbbomb++;
                        }
}

void Game::fire_col_bomb_right()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    std::vector<Bomb>::iterator it2 = _bomb.begin();

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (it2 = _bomb.begin(); it2 != _bomb.end(); ++it2)
                if (it2->_state == 'a')
                    for (int i = 0; i < it->radius[2]; i++)
                        if (it2->_pos.x == it->_pos.x + i && it2->_pos.z == it->_pos.z) {
                            it2->_state = 'b';
                            it2->_tcreated = clock();
                            it2->radius[0] = radius_left(*it2);
                            it2->radius[1] = radius_up(*it2);
                            it2->radius[2] = radius_right(*it2);
                            it2->radius[3] = radius_down(*it2);
                            it2->character->_nbbomb++;
                        }
}

void Game::fire_col_bomb_up()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    std::vector<Bomb>::iterator it2 = _bomb.begin();

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (it2 = _bomb.begin(); it2 != _bomb.end(); ++it2)
                if (it2->_state == 'a')
                    for (int i = 0; i < it->radius[1]; i++)
                        if (it2->_pos.x == it->_pos.x && it2->_pos.z == it->_pos.z - i) {
                            it2->_state = 'b';
                            it2->_tcreated = clock();
                            it2->radius[0] = radius_left(*it2);
                            it2->radius[1] = radius_up(*it2);
                            it2->radius[2] = radius_right(*it2);
                            it2->radius[3] = radius_down(*it2);
                            it2->character->_nbbomb++;
                        }
}

void Game::fire_col_bomb_down()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    std::vector<Bomb>::iterator it2 = _bomb.begin();

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (it2 = _bomb.begin(); it2 != _bomb.end(); ++it2)
                if (it2->_state == 'a')
                    for (int i = 0; i < it->radius[3]; i++)
                        if (it2->_pos.x == it->_pos.x && it2->_pos.z == it->_pos.z + i) {
                            it2->_state = 'b';
                            it2->_tcreated = clock();
                            it2->radius[0] = radius_left(*it2);
                            it2->radius[1] = radius_up(*it2);
                            it2->radius[2] = radius_right(*it2);
                            it2->radius[3] = radius_down(*it2);
                            it2->character->_nbbomb++;
                        }
}

void Game::fire_col_bomb() // collision entre une explosion et une bombe et la fait exploser
{
    fire_col_bomb_left();
    fire_col_bomb_right();
    fire_col_bomb_up();
    fire_col_bomb_down();
}

void Game::fire_col_player_left()
{
    std::vector<Bomb>::iterator it = _bomb.begin();

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (int i = 0; i < it->radius[0]; i++)
                for (int y = 0; y < _nb_player + _nb_ia; y++)
                    if (_character[y].character_pos.x > it->_pos.x - 0.50f - i &&
                    _character[y].character_pos.x < it->_pos.x + 0.50f - i &&
                    _character[y].character_pos.z > it->_pos.z - 0.50f &&
                    _character[y].character_pos.z < it->_pos.z + 0.50f && _character[y]._live > 0) {
                        PlaySoundMulti(_sound[2]);
                        _character[y]._live = 0;
                    }
}

void Game::fire_col_player_right()
{
    std::vector<Bomb>::iterator it = _bomb.begin();

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (int i = 0; i < it->radius[2]; i++)
                for (int y = 0; y < _nb_player + _nb_ia; y++)
                    if (_character[y].character_pos.x > it->_pos.x - 0.50f + i &&
                    _character[y].character_pos.x < it->_pos.x + 0.50f + i &&
                    _character[y].character_pos.z > it->_pos.z - 0.50f &&
                    _character[y].character_pos.z < it->_pos.z + 0.50f && _character[y]._live > 0) {
                        PlaySoundMulti(_sound[2]);
                        _character[y]._live = 0;
                    }
}

void Game::fire_col_player_up()
{
    std::vector<Bomb>::iterator it = _bomb.begin();

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (int i = 0; i < it->radius[1]; i++)
                for (int y = 0; y < _nb_player + _nb_ia; y++)
                    if (_character[y].character_pos.x > it->_pos.x - 0.50f &&
                    _character[y].character_pos.x < it->_pos.x + 0.50f &&
                    _character[y].character_pos.z > it->_pos.z - 0.50f -i &&
                    _character[y].character_pos.z < it->_pos.z + 0.50f - i && _character[y]._live > 0) {
                        PlaySoundMulti(_sound[2]);
                        _character[y]._live = 0;
                    }
}

void Game::fire_col_player_down()
{
    std::vector<Bomb>::iterator it = _bomb.begin();

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (int i = 0; i < it->radius[3]; i++)
                for (int y = 0; y < _nb_player + _nb_ia; y++)
                    if (_character[y].character_pos.x > it->_pos.x - 0.50f &&
                    _character[y].character_pos.x < it->_pos.x + 0.50f &&
                    _character[y].character_pos.z > it->_pos.z - 0.50f + i &&
                    _character[y].character_pos.z < it->_pos.z + 0.50f + i && _character[y]._live > 0) {
                        PlaySoundMulti(_sound[2]);
                        _character[y]._live = 0;
                    }
}

void Game::fire_col_player() //colision entre une explosion et un joueur et le tue
{
    fire_col_player_left();
    fire_col_player_right();
    fire_col_player_up();
    fire_col_player_down();
}

void Game::fire_col_bonus_left()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    std::vector<Bonus>::iterator it2 = _bonus.begin();
    int time;

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (int i = 0; i < it->radius[0]; i++)
                for (it2 = _bonus.begin(); it2 != _bonus.end(); it2++)
                    if (it2->_pos.x == it->_pos.x - i && it2->_pos.z == it->_pos.z && (time = (clock() - it2->_tcreated) * 1000 / CLOCKS_PER_SEC) > 300) {
                        _bonus.erase(it2);
                        --it2;
                    }
}

void Game::fire_col_bonus_right()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    std::vector<Bonus>::iterator it2 = _bonus.begin();
    int time;

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (int i = 0; i < it->radius[0]; i++)
                for (it2 = _bonus.begin(); it2 != _bonus.end(); it2++)
                    if (it2->_pos.x == it->_pos.x + i && it2->_pos.z == it->_pos.z && (time = (clock() - it2->_tcreated) * 1000 / CLOCKS_PER_SEC) > 300) {
                        _bonus.erase(it2);
                        --it2;
                    }
}

void Game::fire_col_bonus_up()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    std::vector<Bonus>::iterator it2 = _bonus.begin();
    int time;

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (int i = 0; i < it->radius[0]; i++)
                for (it2 = _bonus.begin(); it2 != _bonus.end(); it2++)
                    if (it2->_pos.x == it->_pos.x && it2->_pos.z == it->_pos.z - i && (time = (clock() - it2->_tcreated) * 1000 / CLOCKS_PER_SEC) > 300) {
                        _bonus.erase(it2);
                        --it2;
                    }
}

void Game::fire_col_bonus_down()
{
    std::vector<Bomb>::iterator it = _bomb.begin();
    std::vector<Bonus>::iterator it2 = _bonus.begin();
    int time;

    for (it = _bomb.begin(); it != _bomb.end(); ++it)
        if (it->_state == 'b')
            for (int i = 0; i < it->radius[0]; i++)
                for (it2 = _bonus.begin(); it2 != _bonus.end(); it2++)
                    if (it2->_pos.x == it->_pos.x && it2->_pos.z == it->_pos.z + i && (time = (clock() - it2->_tcreated) * 1000 / CLOCKS_PER_SEC) > 300) {
                        _bonus.erase(it2);
                        --it2;
                    }
}

void Game::fire_col_bonus()
{
    fire_col_bonus_up();
    fire_col_bonus_down();
    fire_col_bonus_right();
    fire_col_bonus_left();
}