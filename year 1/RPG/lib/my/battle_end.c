/*
** EPITECH PROJECT, 2020
** battle_end
** File description:
** about end battle
*/

#include "../../include/mylist.h"

void draw_over(battle_t *menu, player_t *player)
{
    sfVector2f posa = { 960, 340 };
    sfVector2f posb = { 960, 550 };

    if (player->pv <= 0) {
        draw_leave(menu, 0);
        if (menu->mob[0]->rect.top != 363 && menu->mob[0]->rect.top != 726) {
        menu->mob[0]->rect.left = 726;
        menu->mob[0]->rect.top = 363;
        }
        mvbt_battle(menu->button, -2000, 0);
        sfRectangleShape_setPosition(menu->button[19]->sprite, posa);
        sfRectangleShape_setPosition(menu->button[20]->sprite, posb);
        sfText_setPosition(menu->button[19]->text, posa);
        sfText_setPosition(menu->button[20]->text, posb);
    }
}

void draw_win2(battle_t *menu, player_t *player, int loot, int looted)
{
    if (loot == 1 && looted == 0) {
        sfText_setString(menu->button[22]->text, "You get a mana potion");
        player->nbpotmana += 1;
        mvbt_battle(menu->button, -2000, 0);
    }
    if (loot == 2 && looted == 0) {
        sfText_setString(menu->button[22]->text, "You get a heal potion");
        player->nbpothp += 1;
        mvbt_battle(menu->button, -2000, 0);
    }
    if (loot == 3 && looted == 0) {
        sfText_setString(menu->button[22]->text, "You get a fire potion");
        player->nbpotfire += 1;
        mvbt_battle(menu->button, -2000, 0);
    }
}

void draw_win(battle_t *menu, player_t *player)
{
    sfVector2f posa = { 960, 300 };
    sfVector2f posb = { 960, 500 };
    sfVector2f posc = { 960, 650 };
    static int looted = 0;
    int loot = rand() % 3 + 1;

    if (menu->mob[1]->pv > 0)
        looted = 0;
    if (menu->mob[1]->pv <= 0 && looted == 0) {
        draw_leave(menu, 0);
        draw_win2(menu, player, loot, looted);
        sfRectangleShape_setPosition(menu->button[21]->sprite, posa);
        sfRectangleShape_setPosition(menu->button[22]->sprite, posb);
        sfRectangleShape_setPosition(menu->button[23]->sprite, posc);
        sfText_setPosition(menu->button[21]->text, posa);
        sfText_setPosition(menu->button[22]->text, posb);
        sfText_setPosition(menu->button[23]->text, posc);
        looted = 1;
    }
}