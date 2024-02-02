#include "graphic.hpp"

#define MAXX 13
#define MAXY 11

char wallGeneration(int i, int j)
{
    if (i % 2 != 0 && j % 2 != 0)
        return 1;
    return 0;
}

char blockGeneration(int i, int j, int sizeX, int sizeY)
{
    int rnd = std::rand() % 100;

    if (rnd < 30)
        return ' ';
    else if (rnd >= 30 &&  rnd < 100)
        return '&';
    return '&';
}

char Generation(int i, int j, int sizeX, int sizeY)
{
    if (wallGeneration(i, j))
        return '*';
    return blockGeneration(i, j, sizeX, sizeY);
}

void spawnPoint(std::map<Vector2, char> &map, float sizeX, float sizeY)
{
    map[Vector2{0, 0}] = ' ';
    map[Vector2{1, 0}] = ' ';
    map[Vector2{0, 1}] = ' ';
    map[Vector2{0, 2}] = '&';
    map[Vector2{2, 0}] = '&';

    map[Vector2{sizeX - 1, 0}] = ' ';
    map[Vector2{sizeX - 1, 1}] = ' ';
    map[Vector2{sizeX - 2, 0}] = ' ';
    map[Vector2{sizeX - 1, 2}] = '&';
    map[Vector2{sizeX - 3, 0}] = '&';

    map[Vector2{0, sizeY - 1}] = ' ';
    map[Vector2{0, sizeY - 2}] = ' ';
    map[Vector2{1, sizeY - 1}] = ' ';
    map[Vector2{0, sizeY - 3}] = '&';
    map[Vector2{2, sizeY - 3}] = '&';

    map[Vector2{sizeX - 1, sizeY - 1}] = ' ';
    map[Vector2{sizeX - 1, sizeY - 2}] = ' ';
    map[Vector2{sizeX - 2, sizeY - 1}] = ' ';
    map[Vector2{sizeX - 1, sizeY - 3}] = '&';
    map[Vector2{sizeX - 3, sizeY - 1}] = '&';
}

std::map<Vector2, char> createMap()
{
    std::map<Vector2, char> map;

    for (int y = 0; y < MAXY; ++y) {
        for (int x = 0; x < MAXX; ++x)
        map.emplace(Vector2{(float)x, (float)y}, Generation(x, y, MAXX, MAXY));
    }
    spawnPoint(map, MAXX, MAXY);
    return map;
}