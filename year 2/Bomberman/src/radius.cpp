#include "graphic.hpp"

int Game::radius_left(Bomb bomb) //permet de determiner la range d'une explosion avec les collisions des cubes
{                                // detruit les obstacles et fait pop un bonus
    std::map<Vector2,char>::iterator it;
    int radius = 0;
    int random;

    for (int i = 0; i < bomb._power; i++) {
        it = _map.find({bomb._pos.x - i, bomb._pos.z});
        if (it->second == '*' || it->second == 'b')
            break;
        if (it->second == ' ' || it->second == 'X' || it->second == 'o' || it->second == '!')
            ++radius;
        if (it->second == '&') {
            ++radius;
            it->second = 'X';
            random = rand() % 7;
            if (random > 0 && random < 5)
                _bonus.push_back(Bonus(it->first.x, it->first.y, random + 48));
            break;
        }
    }
    return (radius);
}

int Game::radius_right(Bomb bomb)
{
    std::map<Vector2,char>::iterator it;
    int radius = 0;
    int random;

    for (int i = 0; i < bomb._power; i++) {
        it = _map.find({bomb._pos.x + i, bomb._pos.z});
        if (it->second == '*' || it->second == 'b')
            break;
        if (it->second == ' ' || it->second == 'X' || it->second == 'o' || it->second == '!')
            ++radius;
        if (it->second == '&') {
            ++radius;
            it->second = 'X';
            random = rand() % 7;
            if (random > 0 && random < 5)
                _bonus.push_back(Bonus(it->first.x, it->first.y, random + 48));
            break;
        }
    }
    return (radius);
}

int Game::radius_up(Bomb bomb)
{
    std::map<Vector2,char>::iterator it;
    int radius = 0;
    int random;

    for (int i = 0; i < bomb._power; i++) {
        it = _map.find({bomb._pos.x, bomb._pos.z - i});
        if (it->second == '*' || it->second == 'b')
            break;
        if (it->second == ' ' || it->second == 'X' || it->second == 'o' || it->second == '!')
            ++radius;
        if (it->second == '&') {
            ++radius;
            it->second = 'X';
            random = rand() % 7;
            if (random > 0 && random < 5)
                _bonus.push_back(Bonus(it->first.x, it->first.y, random + 48));
            break;
        }
    }
    return (radius);
}

int Game::radius_down(Bomb bomb)
{
    std::map<Vector2,char>::iterator it;
    int radius = 0;
    int random;

    for (int i = 0; i < bomb._power; i++) {
        it = _map.find({bomb._pos.x, bomb._pos.z + i});
        if (it->second == '*' || it->second == 'b')
            break;
        if (it->second == ' ' || it->second == 'X' || it->second == 'o' || it->second == '!')
            ++radius;
        if (it->second == '&') {
            ++radius;
            it->second = 'X';
            random = rand() % 7;
            if (random > 0 && random < 5)
                _bonus.push_back(Bonus(it->first.x, it->first.y, random + 48));
            break;
        }
    }
    return (radius);
}

void Game::draw_radius_left(Bomb bomb, char data) //permet de determiner la range d'une explosion avec les collisions des cubes
{                                // detruit les obstacles et fait pop un bonus
    std::map<Vector2,char>::iterator it;

    for (int i = 0; i < bomb._power; i++) {
        it = _map.find({bomb._pos.x - i, bomb._pos.z});
        if (it->second == '*' || it->second == 'b' || it->second == '&')
            break;
        if (it->second == ' ' || it->second == 'X' || it->second == '!')
            it->second = data;
    }
}

void Game::draw_radius_right(Bomb bomb, char data)
{
    std::map<Vector2,char>::iterator it;

    for (int i = 0; i < bomb._power; i++) {
        it = _map.find({bomb._pos.x + i, bomb._pos.z});
        if (it->second == '*' || it->second == 'b' || it->second == '&')
            break;
        if (it->second == ' ' || it->second == 'X' || it->second == '!')
            it->second = data;
    }
}

void Game::draw_radius_up(Bomb bomb, char data)
{
    std::map<Vector2,char>::iterator it;

    for (int i = 0; i < bomb._power; i++) {
        it = _map.find({bomb._pos.x, bomb._pos.z - i});
        if (it->second == '*' || it->second == 'b' || it->second == '&')
            break;
        if (it->second == ' ' || it->second == 'X' || it->second == '!')
            it->second = data;
    }
}

void Game::draw_radius_down(Bomb bomb, char data)
{
    std::map<Vector2,char>::iterator it;
    int radius = 0;

    for (int i = 0; i < bomb._power; i++) {
        it = _map.find({bomb._pos.x, bomb._pos.z + i});
        if (it->second == '*' || it->second == 'b' || it->second == '&')
            break;
        if (it->second == ' ' || it->second == 'X' || it->second == '!')
            it->second = data;
    }
}