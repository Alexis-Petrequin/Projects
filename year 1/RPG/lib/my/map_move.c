/*
** EPITECH PROJECT, 2019
** map_move
** File description:
** about move in map
*/

#include "../../include/mylist.h"

static int move_up(sfRenderWindow *window, map_t *map, vector4i_t move)
{
    if (sfKeyboard_isKeyPressed(sfKeyZ)) {
        if (map->perso->rect.top != 0)
            map->perso->rect.left = 0;
        map->perso->rect.top = 0;
        if (!move.up) {
            animation(map);
            sfSprite_move(map->perso->perso, (sfVector2f){ 0, -20 });
        }
        sfSprite_setTextureRect(map->perso->perso, map->perso->rect);
        return (1);
    }
    return (0);
}

static int move_down(sfRenderWindow *window, map_t *map, vector4i_t move)
{
    if (sfKeyboard_isKeyPressed(sfKeyS)) {
        if (map->perso->rect.top != 516)
            map->perso->rect.left = 0;
        map->perso->rect.top = 516;
        if (!move.down) {
            animation(map);
            sfSprite_move(map->perso->perso, (sfVector2f){ 0, 20 });
        }
        sfSprite_setTextureRect(map->perso->perso, map->perso->rect);
        return (1);
    }
    return (0);
}

static int move_right(sfRenderWindow *window, map_t *map, vector4i_t move)
{
    if (sfKeyboard_isKeyPressed(sfKeyD)) {
        if (map->perso->rect.top != 1548)
            map->perso->rect.left = 0;
        map->perso->rect.top = 1548;
        if (!move.right) {
            animation(map);
            sfSprite_move(map->perso->perso, (sfVector2f){ 20, 0 });
        }
        sfSprite_setTextureRect(map->perso->perso, map->perso->rect);
        return (1);
    }
    return (0);
}

static int move_left(sfRenderWindow *window, map_t *map, vector4i_t move)
{
    if (sfKeyboard_isKeyPressed(sfKeyQ)) {
        if (map->perso->rect.top != 1032)
            map->perso->rect.left = 0;
        map->perso->rect.top = 1032;
        if (!move.left) {
            animation(map);
            sfSprite_move(map->perso->perso, (sfVector2f){-20, 0});
        }
        sfSprite_setTextureRect(map->perso->perso, map->perso->rect);
        return (1);
    }
    return (0);
}

int move_gestion(sfRenderWindow *window, map_t *map, vector4i_t move)
{
    if (move_up(window, map, move))
        return (aggro(map));
    if (move_down(window, map, move))
        return (aggro(map));
    if (move_left(window, map, move))
        return (aggro(map));
    if (move_right(window, map, move))
        return (aggro(map));
    map->perso->rect.left = 0;
    sfSprite_setTextureRect(map->perso->perso, map->perso->rect);
    return (0);
}