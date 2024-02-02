/*
** EPITECH PROJECT, 2020
** battle_draw2
** File description:
** about battle draw
*/

#include "../../include/mylist.h"

void update_player(player_t *player)
{
    player->pvmax = 85 + (player->level * 15);
    player->manamax = 45 + (player->level * 5);
    player->force = 10 + (player->level * 2);
    player->defense = 2 + (player->level * 2);
    player->magic = 1 + (player->level);
    player->mana = player->manamax;
    player->pv = player->pvmax;
}

void update_stats(battle_t *m, player_t *player)
{
    char str[5] = { "\0\0\0\0\0" };

    sfText_setString(m->button[13]->text, numtostr(m->mob[1]->pv, str));
    sfText_setString(m->button[14]->text, numtostr(m->mob[1]->mana, str));
    sfText_setString(m->button[15]->text, numtostr(player->pv, str));
    sfText_setString(m->button[16]->text, numtostr(player->mana, str));
}

void lvl_up(player_t *player)
{
    player->monsterkilled += 1;
    if (player->monsterkilled > player->level * 2 && player->level < 10)
        player->level++;
}

int draw_leave(battle_t *menu, int bool)
{
    sfVector2f pos[4] = { {960, 300}, {960, 500}, {960, 650},
                        {-1000, -1000} };
    sfTime time = sfClock_getElapsedTime(menu->clock[3]);

    if (bool == 1 && time.microseconds / 1000000.0 > 3.99)
        for (int i = 24; i <= 26; i++) {
            sfRectangleShape_setPosition(menu->button[i]->sprite,
            pos[i - 24]);
            sfText_setPosition(menu->button[i]->text, pos[i - 24]);
        }
    if (bool == 0)
        for (int i = 24; i <= 26; i++) {
            sfRectangleShape_setPosition(menu->button[i]->sprite, pos[3]);
            sfText_setPosition(menu->button[i]->text, pos[3]);
        }
    return (0);
}

int draw_b(sfRenderWindow *win, battle_t *m, player_t *pl, sfUint8 *buff)
{
    sfTime time = sfClock_getElapsedTime(m->clock[0]);

    shine_all(m, time);
    update_stats(m, pl);
    draw_idle(win, m);
    for (int i = 0; m->sprite[i] != NULL; i++)
        sfRenderWindow_drawSprite(win, m->sprite[i]->sprite, NULL);
    for (int i = 0; m->button[i] != NULL; i++) {
        sfRenderWindow_drawRectangleShape(win, m->button[i]->sprite, NULL);
        sfRenderWindow_drawText(win, m->button[i]->text, NULL);
    }
    for (int i = 0; m->mob[i] != NULL; i++)
        sfRenderWindow_drawSprite(win, m->mob[i]->sprite, NULL);
    if (m->fire == 1)
        draw_fire(win, buff);
    return (0);
}