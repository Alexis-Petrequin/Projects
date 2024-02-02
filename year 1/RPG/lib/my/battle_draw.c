/*
** EPITECH PROJECT, 2020
** battle_draw
** File description:
** about draw
*/

#include "../../include/mylist.h"

int shine_sprite(sprite_t *sprite, int nb)
{
    sfColor color = sfSprite_getColor(sprite->sprite);

    if (color.a + nb > 255)
        sprite->speed.y = 0;
    if (color.a < 20)
        sprite->speed.y = 1;
    if (sprite->speed.y == 1)
        color.a += nb;
    if (sprite->speed.y == 0)
        color.a -= nb;
    sfSprite_setColor(sprite->sprite, color);
    return (0);
}

void shine_all(battle_t *menu, sfTime time)
{
    if (time.microseconds / 1000000.0 > 0.02) {
        shine_sprite(menu->sprite[4], 4);
        shine_sprite(menu->sprite[6], 2);
        sfClock_restart(menu->clock[0]);
    }
}

void idle_player(battle_t *m)
{
    m->mob[0]->rect.left += 363;
    if (m->mob[0]->rect.top == 0)
        m->mob[0]->rect.left += 363;
    if (m->mob[0]->rect.left >= 5445 && (m->mob[0]->rect.top == 1452 ||
    m->mob[0]->rect.top == 0 || m->mob[0]->rect.top == 1815)) {
        m->mob[0]->rect.left = 0;
        m->mob[0]->rect.top = 1452;
    }
    if (m->mob[0]->rect.left >= 5445 && (m->mob[0]->rect.top == 1452 ||
    m->mob[0]->rect.top == 1089 || m->mob[0]->rect.top == 363)) {
        m->mob[0]->rect.left = 0;
        m->mob[0]->rect.top += 363;
    }
    if (m->mob[0]->rect.left >= 5445 && m->mob[0]->rect.top == 726) {
        m->mob[0]->rect.left = 0;
        m->mob[0]->rect.top -= 363;
    }
    if (m->mob[0]->rect.left <= 363 && m->mob[0]->rect.top == 363)
        m->mob[0]->rect.left = 0;
    sfSprite_setTextureRect(m->mob[0]->sprite, m->mob[0]->rect);
}

void idle_mob(battle_t *m)
{
    sfColor color = sfSprite_getColor(m->mob[1]->sprite);

    m->mob[1]->rect.left += 450;
    if (m->mob[1]->rect.left == 8100)
        m->mob[1]->rect.left = 0;
    if (m->mob[1]->pv <= 0 && color.a > 5) {
        color.a -= 5;
        sfSprite_setColor(m->mob[1]->sprite, color);
    }
    sfSprite_setTextureRect(m->mob[1]->sprite, m->mob[1]->rect);
}

int draw_idle(sfRenderWindow *window, battle_t *m)
{
    sfTime time = sfClock_getElapsedTime(m->clock[1]);

    if (time.microseconds / 1000000.0 > 0.05) {
        idle_player(m);
        idle_mob(m);
        sfClock_restart(m->clock[1]);
    }
    return (0);
}