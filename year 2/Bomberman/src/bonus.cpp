#include "graphic.hpp"

Bonus::Bonus(float x, float y, char type)
{
    _pos = { x, 0.70f, y};
    _type = type;
    _tcreated = clock();
    _up = 0.01f;
}

Bonus::Bonus(float x, float y, char type, int tcreated, float up)
{
    _pos = { x, 0.70f, y};
    _type = type;
    _tcreated = tcreated;
    _up = up;
}

void Game::player_col_bonus() // collision entre le joueur et un bonus et l'active
{
    std::vector<Bonus>::iterator it = _bonus.begin();

    for (it = _bonus.begin(); it != _bonus.end(); ++it)
        for (int i = 0; i < _nb_player + _nb_ia; i++) {
            if (it->_type != 'r' && CheckCollisionBoxes(
                {
                    { _character[i].character_pos.x, _character[i].character_pos.y, _character[i].character_pos.z },
                    { _character[i].character_pos.x, _character[i].character_pos.y, _character[i].character_pos.z }
                },
                {
                    { it->_pos.x - 0.50f, CUBESIZE - 0.50f, it->_pos.z - 0.50f },
                    { it->_pos.x + 0.50f, CUBESIZE + 0.50f, it->_pos.z + 0.50f }
                }
            ) && _character[i]._live > 0) {
                PlaySoundMulti(_sound[1]);
                if (it->_type == '1') {
                    _character[i]._obstacle = false;
                    _character[i]._tobstacle = clock();
                }
                if (it->_type == '2') {
                    _character[i]._speed = 6.00f;
                    _character[i]._tspeed = clock();
                }
                if (it->_type == '3')
                    _character[i]._power += 1;
                if (it->_type == '4')
                    _character[i]._nbbomb += 1;
                _bonus.erase(it);
                if (it != _bonus.begin())
                    --it;
                }
        }
}

bool in_block(std::map<Vector2,char> *map, Character *character) // check si le joueur est dans un block
{
    std::map<Vector2,char>::iterator it = map->find({ (int)(character->character_pos.x + 0.48f) + 0.00f, (int)(character->character_pos.z + 0.48f) + 0.00f });

    if (it->second == '&')
        return (true);
    return (false);
}

void Game::timeout_bonus() // desactive les bonus apres un certain temps
{
    int time = clock();

    for (int i = 0; i < _nb_player + _nb_ia; i++) {
        if ( _character[i]._obstacle == false && (time - _character[i]._tobstacle) * 1000 / CLOCKS_PER_SEC > 3000 && in_block(&_map, &_character[i]) == false)
            _character[i]._obstacle = true;
        if ( _character[i]._speed == 6.00f && (time - _character[i]._tspeed) * 1000 / CLOCKS_PER_SEC > 5000)
            _character[i]._speed = 4.00f;
    }
}