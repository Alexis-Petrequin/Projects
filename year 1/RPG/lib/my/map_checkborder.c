/*
** EPITECH PROJECT, 2019
** map_checkborder
** File description:
** about check border
*/

#include "../../include/mylist.h"

void animation(map_t *map)
{
    map->perso->rect.left += 516;
    if (map->perso->rect.left > 14964)
        map->perso->rect.left = 0;
}

static int check_col(sfVector2f spr, sfVector2f perso, int dir, sprit_t *col)
{
    int y = 0;
    int x = 0;
    sfFloatRect dim[4] = { {-20, 15, 80, 125}, {-40, 90, 255, 280},
    {-40, 110, 315, 315}, {-40, 100, 290, 290} };
    if (dir == 1)
        x = -21;
    if (dir == 2)
        x = 21;
    if (dir == 3)
        y = -21;
    if (dir == 4)
        y = 21;
    if (perso.x + x > spr.x + dim[col->type].left &&
        perso.y + y > spr.y + dim[col->type].top &&
        perso.y + y < spr.y + dim[col->type].height &&
        perso.x + x < spr.x + dim[col->type].width)
        return (1);
    return (0);
}

static vector4i_t colision_sprites(map_t *map, vector4i_t move)
{
    sfVector2f perso = sfView_getCenter(map->camera);
    sfVector2f spritepos = { 0, 0 };

    for (int i = 0; map->scenary[i] != NULL; i++) {
        spritepos = sfSprite_getPosition(map->scenary[i]->sprite);
        if (move.left == 0)
            move.left = check_col(spritepos, perso, 1, map->scenary[i]);
        if (move.right == 0)
            move.right = check_col(spritepos, perso, 2, map->scenary[i]);
        if (move.up == 0)
            move.up = check_col(spritepos, perso, 3, map->scenary[i]);
        if (move.down == 0)
            move.down = check_col(spritepos, perso, 4, map->scenary[i]);
    }
    return (move);
}

static vector4i_t border_check(int x, int y, vector4i_t move)
{
    if (x <= 8000)
        move.left = 1;
    if (x >= 16775 || (y <= 6850 && x > 8950))
        move.right = 1;
    if (y >= 14950)
        move.down = 1;
    if (y < 6450 || (y <= 6850 && x > 8970))
        move.up = 1;
    return (move);
}

int move_map(sfRenderWindow *window, map_t *map, player_t *player)
{
    vector4i_t move = { 0, 0, 0, 0 };
    sfVector2f pos = { 0, 0 };
    sfTime time = sfClock_getElapsedTime(map->clock[1]);
    int aggro = 0;

    if (time.microseconds / 1000000.0 > 0.04) {
        pos = sfSprite_getPosition(map->perso->perso);
        move = border_check(pos.x, pos.y, move);
        move = colision_sprites(map, move);
        if (player->state == 4)
            aggro = move_gestion(window, map, move);
        pos = sfSprite_getPosition(map->perso->perso);
        pos.x += 45;
        pos.y += 45;
        sfView_setCenter(map->camera, pos);
        sfRenderWindow_setView(window, map->camera);
        sfClock_restart(map->clock[1]);
    }
    return (aggro);
}