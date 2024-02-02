#include "graphic.hpp"

Vector2 max_border(std::map<Vector2, char> *map) //determine la taille de la map
{
    std::map<Vector2,char>::iterator it = map->begin();
    Vector2 max = {0, 0};

    for (it = map->begin(); it != map->end(); ++it) {
        if (it->first.y > max.y)
            max.y = it->first.y;
        if (it->first.x > max.x)
            max.x = it->first.x;
    }
    return (max);
}

void add_border(std::map<Vector2, char> *map) //ajouter les bords a la map
{
    std::map<Vector2,char>::iterator it = map->begin();
    Vector2 max = max_border(map);

    for (float i = -1.00f; i <= max.x + 1.00f; i += 1.00f) {
        map->insert(std::pair<Vector2, char>({ i, -1.00f }, 'b'));
        map->insert(std::pair<Vector2, char>({ i, max.y + 1.00f }, 'b'));
    }
    for (float i = 0.00f; i <= max.y; i += 1.00f) {
        map->insert(std::pair<Vector2, char>({ -1.00f, i }, 'b'));
        map->insert(std::pair<Vector2, char>({ max.x + 1.00f, i }, 'b'));
    }
}