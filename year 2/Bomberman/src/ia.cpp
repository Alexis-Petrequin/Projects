#include "graphic.hpp"

void Game::IA_should_up(Character *charact)
{
    float x = (int)(charact->character_pos.x + 0.50f) + 0.00f;
    float z = (int)(charact->character_pos.z + 0.50f) + 0.00f;

    charact->_Ia_goal = { x, z - 1.00f };
    charact->_key_up = 1;
}

void Game::IA_should_down(Character *charact)
{
    float x = (int)(charact->character_pos.x + 0.50f) + 0.00f;
    float z = (int)(charact->character_pos.z + 0.50f) + 0.00f;

    charact->_Ia_goal = { x, z + 1.00f };
    charact->_key_down = 1;
}

void Game::IA_should_right(Character *charact)
{
    float x = (int)(charact->character_pos.x + 0.50f) + 0.00f;
    float z = (int)(charact->character_pos.z + 0.50f) + 0.00f;

    charact->_Ia_goal = { x + 1.00f, z };
    charact->_key_right = 1;
}

void Game::IA_should_left(Character *charact)
{
    float x = (int)(charact->character_pos.x + 0.50f) + 0.00f;
    float z = (int)(charact->character_pos.z + 0.50f) + 0.00f;

    charact->_Ia_goal = { x - 1.00f, z };
    charact->_key_left = 1;
}

void Game::IA_move()
{
    for (int i = _nb_player; i < _nb_player + _nb_ia; i++) {
        if (_character[i]._key_up == 1) {
            if (_character[i].character_pos.z > _character[i]._Ia_goal.y && player_col_cube_up(&_character[i], &_map) == false && player_col_bomb_up(&_character[i], &_bomb) == false) {
                _character[i].model.transform = MatrixRotateXYZ({ 0.0f, 0.0f, 135.0f });
                _character[i].character_pos.z -= GetFrameTime() * _character[i]._speed;
                UpdateModelAnimation(_character[i].model, _character[i].anims[0], _character[i].animFrameCounter++);
            } else {
                _character[i]._key_up = 0;
                _character[i]._Ia_goal = { -1.00f, -1.00f};
            }
            continue;
        }
        if (_character[i]._key_down == 1) {
            if (_character[i].character_pos.z < _character[i]._Ia_goal.y && player_col_cube_down(&_character[i], &_map) == false && player_col_bomb_down(&_character[i], &_bomb) == false) {
                _character[i].model.transform = MatrixRotateXYZ({ 0.0f, 0.0f, 270.0f });
                _character[i].character_pos.z += GetFrameTime() * _character[i]._speed;
                UpdateModelAnimation(_character[i].model, _character[i].anims[0], _character[i].animFrameCounter++);
            } else {
                _character[i]._key_down = 0;
                _character[i]._Ia_goal = { -1.00f, -1.00f};
            }
            continue;
        }
        if (_character[i]._key_left == 1) {
            if (_character[i].character_pos.x > _character[i]._Ia_goal.x && player_col_cube_left(&_character[i], &_map) == false && player_col_bomb_left(&_character[i], &_bomb) == false) {
                _character[i].model.transform = MatrixRotateXYZ({ 0.0f, 0.0f, 45.0f });
                _character[i].character_pos.x -= GetFrameTime() * _character[i]._speed;
                UpdateModelAnimation(_character[i].model, _character[i].anims[0], _character[i].animFrameCounter++);
            } else {
                _character[i]._key_left = 0;
                _character[i]._Ia_goal = { -1.00f, -1.00f};
            }
            continue;
        }
        if (_character[i]._key_right == 1) {
            if (_character[i].character_pos.x < _character[i]._Ia_goal.x && player_col_cube_right(&_character[i], &_map) == false && player_col_bomb_right(&_character[i], &_bomb) == false) {
                _character[i].model.transform = MatrixRotateXYZ({ 0.0f, 0.0f, 180.0f });
                _character[i].character_pos.x += GetFrameTime() * _character[i]._speed;
                UpdateModelAnimation(_character[i].model, _character[i].anims[0], _character[i].animFrameCounter++);
            } else {
                _character[i]._key_right = 0;
                _character[i]._Ia_goal = { -1.00f, -1.00f};
            }
            continue;
        }
        _character[i].animFrameCounter = 0;
        UpdateModelAnimation(_character[i].model, _character[i].anims[0], _character[i].animFrameCounter);
    }
}

int Game::dodge_right(Vector2 pos)
{
    std::map<Vector2,char>::iterator it = _map.find({pos.x + 1.00f, pos.y});

    if (it->second == ' ')
        return (1);
    if (it->second == '&' || it->second == '*' || it->second == 'b' || it->second == 'o' || it->second == '!')
        return (-1);
    if (it->second == 'X')
        return (0);
    return (-1);
}

int Game::dodge_left(Vector2 pos)
{
    std::map<Vector2,char>::iterator it = _map.find({pos.x - 1.00f, pos.y});

    if (it->second == ' ')
        return (1);
    if (it->second == '&' || it->second == '*' || it->second == 'b' || it->second == 'o' || it->second == '!')
        return (-1);
    if (it->second == 'X')
        return (0);
    return (-1);
}

int Game::dodge_up(Vector2 pos)
{
    std::map<Vector2,char>::iterator it = _map.find({pos.x, pos.y - 1.00f});

    if (it->second == ' ')
        return (1);
    if (it->second == '&' || it->second == '*' || it->second == 'b' || it->second == 'o' || it->second == '!')
        return (-1);
    if (it->second == 'X')
        return (0);
    return (-1);
}

int Game::dodge_down(Vector2 pos)
{
    std::map<Vector2,char>::iterator it = _map.find({pos.x, pos.y + 1.00f});

    if (it->second == ' ')
        return (1);
    if (it->second == '&' || it->second == '*' || it->second == 'b' || it->second == 'o' || it->second == '!')
        return (-1);
    if (it->second == 'X')
        return (0);
    return (-1);
}

int Game::dodge_bomb(Character *IA)
{
    float posx = (int)(IA->character_pos.x + 0.50f) + 0.00f;
    float posy = (int)(IA->character_pos.z + 0.50f) + 0.00f;
    int up = 0;
    int down = 0;
    int left = 0;
    int right = 0;

    if (_map.find({posx, posy})->second != 'X' && _map.find({posx, posy})->second != 'o')
        return (-1);
    down = dodge_down({ posx, posy });
    if (down == 1) {
        IA_should_down(IA);
        return (1);
    }
    up = dodge_up({ posx, posy });
    if (up == 1) {
        IA_should_up(IA);
        return (1);
    }
    left = dodge_left({ posx, posy });
    if (left == 1) {
        IA_should_left(IA);
        return (1);
    }
    right = dodge_right({ posx, posy });
    if (right == 1) {
        IA_should_right(IA);
        return (1);
    }
    for (int i = 1; i < 5; i++) {
        if (left != -1) {
            if (dodge_down({ posx - (1.00f * i), posy }) == 1) {
                IA_should_left(IA);
                return (1);
            }
            if (dodge_up({ posx - (1.00f * i), posy }) == 1) {
                IA_should_left(IA);
                return (1);
            }
            left = dodge_left({ posx - (1.00f * i), posy });
            if (left == 1) {
                IA_should_left(IA);
                return (1);
            }
        }
        if (right != -1) {
            if (dodge_down({ posx + (1.00f * i), posy }) == 1) {
                IA_should_right(IA);
                return (1);
            }
            if (dodge_up({ posx + (1.00f * i), posy }) == 1) {
                IA_should_right(IA);
                return (1);
            }
            right = dodge_right({ posx + (1.00f * i), posy });
            if (right == 1) {
                IA_should_right(IA);
                return (1);
            }
        }
        if (up != -1) {
            if (dodge_left({ posx, posy - (1.00f * i) }) == 1) {
                IA_should_up(IA);
                return (1);
            }
            if (dodge_right({ posx, posy - (1.00f * i) }) == 1) {
                IA_should_up(IA);
                return (1);
            }
            up = dodge_up({ posx, posy - (1.00f * i) });
            if (up == 1) {
                IA_should_up(IA);
                return (1);
            }
        }
        if (down != -1) {
            if (dodge_left({ posx, posy + (1.00f * i) }) == 1) {
                IA_should_down(IA);
                return (1);
            }
            if (dodge_right({ posx, posy + (1.00f * i) }) == 1) {
                IA_should_down(IA);
                return (1);
            }
            down = dodge_down({ posx, posy + (1.00f * i) });
            if (down == 1) {
                IA_should_down(IA);
                return (1);
            }
        }
    }
    return (0);
}

Vector2 Game::Ia_select_dest(Character *IA, char data)
{
    std::map<Vector2,char>::iterator it = _map.begin();
    Vector2 pos = { -1.00f, -1.00f };
    int posx = (int)(IA->character_pos.x + 0.50f);
    int posy = (int)(IA->character_pos.z + 0.50f);
    int distance = 0;
    int closest = 1000;

    for (it = _map.begin(); it != _map.end(); ++it)
        if (it->second == data) {
            distance = abs((int)it->first.x - posx) + abs((int)it->first.y - posy);
            if (distance < closest) {
                closest = distance;
                pos.x = it->first.x;
                pos.y = it->first.y;
            }
        }
    return (pos);
}

int Game::IA_go_box(Character *IA)
{
    Vector2 dest = Ia_select_dest(IA, '&');
    float posx = (int)(IA->character_pos.x + 0.50f) + 0.00f;
    float posy = (int)(IA->character_pos.z + 0.50f) + 0.00f;
    int way = rand() % 2;

    if (dest.x == -1.00f && dest.y == -1.00f)
        return (-1);
    if (way == 1) {
        if (posx != dest.x) {
            if (posx < dest.x && _map.find({posx + 1, posy})->second == ' ') {
                IA_should_right(IA);
                return (1);
            }
            if (posx > dest.x && _map.find({posx - 1, posy})->second == ' ') {
                IA_should_left(IA);
                return (1);
            }
        }
        if (posy != dest.y) {
                if (posy < dest.y && _map.find({posx, posy + 1})->second == ' ') {
                    IA_should_down(IA);
                    return (1);
                }
                if (posy > dest.y && _map.find({posx, posy - 1})->second == ' ') {
                    IA_should_up(IA);
                    return (1);
                }
        }
        if (posy != dest.y) {
            if (way == 1 && _map.find({posx + 1, posy})->second == ' ') {
                IA_should_right(IA);
                return (1);
        }
            if (way == 0 && _map.find({posx - 1, posy})->second == ' ') {
                IA_should_left(IA);
                return (1);
            }
        }
        if (posx != dest.x) {
            if (way == 1 && _map.find({posx, posy + 1})->second == ' ') {
                IA_should_down(IA);
                return (1);
            }
            if (way == 0 && _map.find({posx, posy - 1})->second == ' ') {
                IA_should_up(IA);
                return (1);
            }
        }
    }

    if (way == 0) {
        if (posy != dest.y) {
                if (posy < dest.y && _map.find({posx, posy + 1})->second == ' ') {
                    IA_should_down(IA);
                    return (1);
                }
                if (posy > dest.y && _map.find({posx, posy - 1})->second == ' ') {
                    IA_should_up(IA);
                    return (1);
                }
        }
        if (posx != dest.x) {
            if (posx < dest.x && _map.find({posx + 1, posy})->second == ' ') {
                IA_should_right(IA);
                return (1);
            }
            if (posx > dest.x && _map.find({posx - 1, posy})->second == ' ') {
                IA_should_left(IA);
                return (1);
            }
        }
        if (posx != dest.x) {
            if (way == 1 && _map.find({posx, posy + 1})->second == ' ') {
                IA_should_down(IA);
                return (1);
            }
            if (way == 0 && _map.find({posx, posy - 1})->second == ' ') {
                IA_should_up(IA);
                return (1);
            }
        }
        if (posy != dest.y) {
            if (way == 1 && _map.find({posx + 1, posy})->second == ' ') {
                IA_should_right(IA);
                return (1);
        }
            if (way == 0 && _map.find({posx - 1, posy})->second == ' ') {
                IA_should_left(IA);
                return (1);
            }
        }
    }
    return (0);
}

bool Game::still_box()
{
    std::map<Vector2,char>::iterator it = _map.begin();

    for (it = _map.begin(); it != _map.end(); ++it)
        if (it->second == '&')
            return true;
    return false;
}

bool Game::IA_attack(Character *IA, Character *focus)
{

    float IAposx = (int)(IA->character_pos.x + 0.50f) + 0.00f;
    float IAposy = (int)(IA->character_pos.z + 0.50f) + 0.00f;
    float Fposx = (int)(focus->character_pos.x + 0.50f) + 0.00f;
    float Fposy = (int)(focus->character_pos.z + 0.50f) + 0.00f;
    int distance = 0;
    char data;

    if ( IAposx > Fposx) {
        distance = (int)(IAposx - Fposx);
        if ((IA->_power - 1) >= distance) {
            for (int i = 0; i < distance; i++) {
                data = _map.find({IAposx - i, IAposy})->second;
                if (data != ' ' && data != 'X' && data != '!' && data != 'o')
                    return (false);
            }
            return (true);
        }
    }
    if ( IAposx < Fposx) {
        distance = (int)(Fposx - IAposx);
        if ((IA->_power - 1) >= distance) {
            for (int i = 0; i < distance; i++) {
                data = _map.find({IAposx + i, IAposy})->second;
                if (data != ' ' && data != 'X' && data != '!' && data != 'o')
                    return (false);
            }
        return (true);
        }
    }
    if ( IAposy > Fposy) {
        distance = (int)(IAposy - Fposy);
        if ((IA->_power - 1) >= distance) {
            for (int i = 0; i < distance; i++) {
                data = _map.find({IAposx, IAposy - i})->second;
                if (data != ' ' && data != 'X' && data != '!' && data != 'o')
                    return (false);
            }
        return (true);
        }
    }
    if ( IAposy < Fposy) {
        distance = (int)(Fposy - IAposy);
        if ((IA->_power - 1) >= distance) {
            for (int i = 0; i < distance; i++) {
                data = _map.find({IAposx, IAposy + i})->second;
                if (data != ' ' && data != 'X' && data != '!' && data != 'o')
                    return (false);
            }
        return (true);
        }
    }
    return (false);
}

int Game::IA_put_bomb(Character *IA)
{
    float posx = (int)(IA->character_pos.x + 0.50f) + 0.00f;
    float posy = (int)(IA->character_pos.z + 0.50f) + 0.00f;
    std::map<Vector2,char>::iterator it = _map.find({posx, posy});
    int obst = 0;

    if (IA->_nbbomb == 0 || it->second == 'X' || it->second == 'o' || it->second == '&')
        return (-1);
    if (_map.find({posx + 1.00f, posy})->second == 'X' || _map.find({posx - 1.00f, posy})->second == 'X' ||
    _map.find({posx, posy + 1.00f})->second == 'X' || _map.find({posx, posy - 1.00f})->second == 'X')
        return (-1);
    if (_map.find({posx + 1.00f, posy})->second == '&' || _map.find({posx - 1.00f, posy})->second == '&' ||
    _map.find({posx, posy + 1.00f})->second == '&' || _map.find({posx, posy - 1.00f})->second == '&') {
        Bomb new_bomb(IA->_power, posx, posy, 'a', IA);
        _bomb.push_back(new_bomb);
        IA->_nbbomb--;
        return (1);
    }
    for (int i = 0; i < _nb_ia + _nb_player; i++)
        if (_character[i]._live == 1)
            if ( ((int)(_character[i].character_pos.x + 0.50f) + 0.00f != posx || (int)(_character[i].character_pos.z + 0.50f) + 0.00f != posy) &&
            (((int)(_character[i].character_pos.x + 0.50f) + 0.00f == posx || (int)(_character[i].character_pos.z + 0.50f) + 0.00f == posy)))
                if (IA_attack(IA, &_character[i]) == true) {
                    Bomb new_bomb(IA->_power, posx, posy, 'a', IA);
                    _bomb.push_back(new_bomb);
                    IA->_nbbomb--;
                    return (1);
                }
    return (0);
}

Vector2 Game::Ia_find_player(Character *IA)
{
    float posx = (int)(IA->character_pos.x + 0.50f) + 0.00f;
    float posy = (int)(IA->character_pos.z + 0.50f) + 0.00f;
    Vector2 pos = { -1.00f, -1.00f };
    int distance = 0;
    int closest = 1000;

    for (int i = 0; i < _nb_ia + _nb_player; i++)
        if (_character[i]._live == 1)
            if ( (int)(_character[i].character_pos.x + 0.50f) + 0.00f != posx || (int)(_character[i].character_pos.z + 0.50f) + 0.00f != posy) {
                distance = (int)(abs((int)_character[i].character_pos.x - posx) + abs((int)_character[i].character_pos.z - posy));
                if (distance < closest && distance > 1) {
                    closest = distance;
                    pos.x = _character[i].character_pos.x;
                    pos.y = _character[i].character_pos.z;
                }
            }
    return (pos);
}

int Game::Ia_go_player(Character *IA)
{
    Vector2 dest = Ia_find_player(IA);
    float posx = (int)(IA->character_pos.x + 0.50f) + 0.00f;
    float posy = (int)(IA->character_pos.z + 0.50f) + 0.00f;
    int way = rand() % 2;

    if (dest.x == -1.00f && dest.y == -1.00f)
        return (-1);
    if (way == 1) {
        if (posx != dest.x) {
            if (posx < dest.x && _map.find({posx + 1, posy})->second == ' ') {
                IA_should_right(IA);
                return (1);
            }
            if (posx > dest.x && _map.find({posx - 1, posy})->second == ' ') {
                IA_should_left(IA);
                return (1);
            }
        }
        if (posy != dest.y) {
                if (posy < dest.y && _map.find({posx, posy + 1})->second == ' ') {
                    IA_should_down(IA);
                    return (1);
                }
                if (posy > dest.y && _map.find({posx, posy - 1})->second == ' ') {
                    IA_should_up(IA);
                    return (1);
                }
        }
        if (posy != dest.y) {
            if (way == 1 && _map.find({posx + 1, posy})->second == ' ') {
                IA_should_right(IA);
                return (1);
        }
            if (way == 0 && _map.find({posx - 1, posy})->second == ' ') {
                IA_should_left(IA);
                return (1);
            }
        }
        if (posx != dest.x) {
            if (way == 1 && _map.find({posx, posy + 1})->second == ' ') {
                IA_should_down(IA);
                return (1);
            }
            if (way == 0 && _map.find({posx, posy - 1})->second == ' ') {
                IA_should_up(IA);
                return (1);
            }
        }
    }

    if (way == 0) {
        if (posy != dest.y) {
                if (posy < dest.y && _map.find({posx, posy + 1})->second == ' ') {
                    IA_should_down(IA);
                    return (1);
                }
                if (posy > dest.y && _map.find({posx, posy - 1})->second == ' ') {
                    IA_should_up(IA);
                    return (1);
                }
        }
        if (posx != dest.x) {
            if (posx < dest.x && _map.find({posx + 1, posy})->second == ' ') {
                IA_should_right(IA);
                return (1);
            }
            if (posx > dest.x && _map.find({posx - 1, posy})->second == ' ') {
                IA_should_left(IA);
                return (1);
            }
        }
        if (posx != dest.x) {
            if (way == 1 && _map.find({posx, posy + 1})->second == ' ') {
                IA_should_down(IA);
                return (1);
            }
            if (way == 0 && _map.find({posx, posy - 1})->second == ' ') {
                IA_should_up(IA);
                return (1);
            }
        }
        if (posy != dest.y) {
            if (way == 1 && _map.find({posx + 1, posy})->second == ' ') {
                IA_should_right(IA);
                return (1);
        }
            if (way == 0 && _map.find({posx - 1, posy})->second == ' ') {
                IA_should_left(IA);
                return (1);
            }
        }
    }
    return (0);
}

void Game::Ia_rand_dest(Character *IA)
{
    float posx = (int)(IA->character_pos.x + 0.50f) + 0.00f;
    float posy = (int)(IA->character_pos.z + 0.50f) + 0.00f;
    int way = rand() % 4;

    if (way == 0 && _map.find({posx - 1, posy})->second == ' ') {
        IA_should_left(IA);
        return;
    }
    if (way == 1 && _map.find({posx + 1, posy})->second == ' ') {
        IA_should_right(IA);
        return;
    }
    if (way == 2 && _map.find({posx, posy - 1})->second == ' ') {
        IA_should_up(IA);
        return;
    }
    if (way == 3 && _map.find({posx, posy + 1})->second == ' ') {
        IA_should_down(IA);
        return;
    }
}

void Game::Ia_find_dest(Character *IA)
{
    if (IA_put_bomb(IA) == 1)
        return;
    if (dodge_bomb(IA) != -1)
        return;
    if (still_box()) {
        IA_go_box(IA);
        return;
    }
    if (Ia_go_player(IA) != -1)
        return;
    Ia_rand_dest(IA);
}

void Game::ia()
{
    if (clock() * 1000 / CLOCKS_PER_SEC > 1300) {
        for (int i = _nb_player; i < _nb_player + _nb_ia; i++)
            if (_character[i]._live == 1 && _character[i]._Ia_goal.x == -1.00f && _character[i]._Ia_goal.y == -1.00f)
                Ia_find_dest(&this->_character[i]);
        IA_move();
    }
}